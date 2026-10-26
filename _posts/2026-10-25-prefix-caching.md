---
layout: post
author: Nishant Kelkar
title: Prefix Caching for KV Caches
tags: machine-learning deep-learning system-design
---

Suppose you have a request whose prompt and generated output together span 8,192 tokens. The embedding space of these tokens is `d` dimensional, giving us an `8192 x d` matrix of token embeddings. During both the prefill and decode stages, we would compute many thousands of key and value vectors (KV vectors) across these 8,192 tokens. This blog post goes deeper into how vLLM manages memory for these KV vectors (the so-called "KV Cache") and an optimization that kicks in when prompt prefixes are shared across requests (which in real-world scenarios is always true, e.g. system prompts).


## The Problem

Imagine you have a request spanning 8,192 tokens (prompt + output). Once you project each of these tokens into the $d_{head}$ dimensional space (using the $W_Q$, $W_K$, and $W_V$ projection matrices), we now have a massive computation in front of us.
First, each token's embedding needs to "attend to" all the tokens that came before it. This means each token $t_i$'s query vector $q_i \in \mathbb{R}^{1 \times d_{head}}$ must multiply ${K}^T_{j≤i}$ of dimension $d_{head} \times i$.
Think about the number of values that must be stored for this 1 token $t_i$ so that it is useful in future tokens' attention computation. We need 1 key vector $k_i$, and 1 value vector $v_i$ (both have the same shape). Both $k_i$ and $v_i$ are re-used in future computations for $t_{i+1}$, $t_{i+2}$, etc. to compute the attention these tokens need to pay to the past token $t_i$. The query vector $q_i$ is not required to be stored across Transformer layers simply because future tokens $t_{i+1}$, $t_{i+2}$, etc. never need and use $q_i$.

Thus we end up storing $(2 \times d_{head} \times H \times L)$ floating point values for each token for the attention operation, where $H =$ the number of KV heads per Transformer layer, and $L =$ the number of layers in the model. We'll call the memory needed to hold one token's key and value vectors, across every layer and KV head, a **KV slot** (or just **slot**).

To make this concrete, consider a LlaMA-3-70B model in FP16 quantized format. This model has 80 layers, each layer with 8 KV heads. $d_{head} = 128$, giving us per token `2 x 128 x 8 x 80 = 163,840` floating point values to store!
At FP16 precision, we represent each floating point value with 2 bytes, meaning we need a total of $163,840 \times 2 \approx 328 \text{KB}$. Thus for this model, the KV slot size is 328KB.

With 8,192 tokens, we need a total space of $328 \text{KB} \times 8192 \approx 2.68 \text{GB}$. This is space needed for __one__ request whose prompt and output together fill 8,192 tokens, which happens to be Llama-3-70B's full context window. In the grand scheme of things, this isn't even a long context by modern standards (context lengths as of the time of writing of this post are ~1M with ease).

Now consider the decode stage. To generate each new token, the model attends over every previous token, which means reading the _entire_ KV cache out of GPU memory (HBM): the KV vectors for all the prompt tokens, plus those for every output token generated so far. This cache grows by ~328 KB with each new token, and by the end of our 8,192-token request it holds the full 2.68 GB. Suppose we serve this model on 2xA100 GPUs (80GB each), which together give us about 4 TB/s of memory bandwidth. Reading 2.68 GB at 4 TB/s takes $2.68 \text{ GB} / 4000 \text{ GB/s} \approx 0.7 \text{ ms}$. That is up to ~0.7 ms per output token spent just _moving_ KV vectors, before any math happens.

And the math itself is cheap. Each 256-byte key vector (128 values x 2 bytes) takes part in a single 128-dim dot product with the query, roughly 256 FLOPs, or ~1 FLOP per byte read. An A100 needs ~150 FLOPs per byte (312 TFLOPS / 2 TB/s) before its compute units become the bottleneck. So attention during decode is heavily _memory-bandwidth-bound_: the GPU spends most of its time waiting on memory, not computing.

On its own, 0.7 ms looks small next to the ~35 ms it takes to stream the model's 140 GB of weights each step. But the weights are read once per step for the _whole batch_, while every request has to read its _own_ KV cache. Batching many requests together is how we spread out the cost of reading the weights, which raises the real question: how many requests can we even fit in memory?

The two A100s give us 160GB of HBM2 memory in total, and just holding the model weights requires `70B x 2 bytes = 140 GB`. This leaves 20GB free for other tasks. Assuming CUDA runtime and other driver-level features eat up an additional 10GB, this leaves only 10GB for use with KV slots for requests. If we keep all KV slots in memory per-request, we can hold at most floor(10/2.68) = 3 concurrent requests. Note that this is a __best-case__ estimate - it assumes no memory fragmentation at all from variable-length sequences.

## Solving A Different Problem: Fragmented Memories

The reality is that on a per-request basis, the output sequence length is (a) unknown _a priori_, and (b) can vary from request-to-request. The prompt length, on the other hand, _is_ known the moment a request arrives. Before vLLM came out with their [PagedAttention paper](https://arxiv.org/pdf/2309.06180), serving systems would allocate KV slots to each request based on input prompt length `prompt_length` and a client-supplied `max_tokens` indicating the maximum number of output tokens the request wants to generate. LLM servers restrict the sum total of all tokens seen by the model per-request -- `prompt_length + max_tokens` -- to be limited to at most `max_seq_length`. If `max_tokens` is unsupplied, the number of output tokens the model _may_ generate (i.e. it may still generate fewer tokens by emitting an `<EOS>` early) defaults to `max_seq_length - prompt_length`.

Reserving `prompt_length + max_tokens` KV slots for every request upfront can lead to 3 types of cost:

1. Upfront allocation for the request means that the KV slot for the 100th token out in the future is held right from when the model starts generating the 1st token. Assuming the 100th token _is_ eventually generated, this can be viewed as an "opportunity cost" where this KV slot could have been used for some other concurrent requests.
1. Within the context of a single request, it is likely that the output sequence generation hits a `<EOS>` token and stops when the request holds `T` tokens in total (prompt + output), such that `T < prompt_length + max_tokens`. This means we would have allocated but never used (i.e. wasted) memory worth `(prompt_length + max_tokens - T)` KV slots.
1. Assume that the memory allocator is always allocating KV slots for `prompt_length + max_tokens` tokens per-request. It may allocate this memory contiguously from any region in the HBM address space (assuming this much space is available). Because this reservation varies from request to request, the allocator would run into a problem where a freed segment of memory of `G` KV slots between 2 occupied segments can only serve new requests if they require `≤ G` KV slots, but not for other requests where more than `G` are needed. This is a form of "external" fragmentation wastage where memory remains free because it is too small to support new incoming requests.

To make this more concrete, take the Llama-3-70B model which has a `max_seq_length = 8192`. Assume `max_tokens` is unset. To allocate KV slots for this many tokens, we would need `328 KB (1 KV slot) x 8192 (total tokens) = 2.68 GB` memory. Assume we get a request with a 500-token prompt that ends up generating only 1,500 output tokens though, for 2,000 tokens in total. With this fixed allocation strategy, we would end up using only ~655 MB of this 2.68 GB, effectively wasting ~76% of the allocated KV slots as unused.

PagedAttention removes this hard coupling between upfront KV slot reservation and memory allocation per-request. It takes a page (pun intended) out of Operating Systems design principles, and models memory as blocks of space needed to store `B` KV slots (actual size in bytes is `B x slot_size` where `slot_size` is 328 KB in the above example). In vLLM's experiments, they found that using a block size of 16 KV slots is a good middle ground between the page size being too coarse (leads to per-request wasting, same issue as pre-PagedAttention era allocators) or too fine (leads to GPU kernel inefficiency in the attention implementation as smaller blocks reduce GPU memory-access coalescing and parallelism inside the attention kernel).

Under the hood, PagedAttention achieves this by relying on a block-table. Each request gets its own private block-table. This table maps logical block numbers (e.g. block ids like '0', '1', etc.) to physical block numbers (indices into a large GPU memory pool allocated at startup). Every time a token is generated, the logical block number its KV slot would be stored in is computed (`i`th token's KV slot goes in block `i // B` and in position `i % B` within the block if block size is `B`). That logical block id is looked up in the block-table to get the actual GPU pool index, which is where the KV vectors for this token are actually stored. The block-table also tracks how filled each block is, so that if a block is full, the memory allocator pulls an unused block index from the pool's free list, updates the block table to use it, and resumes generating more tokens.

Standard attention requires KV slots to be in contiguous memory per-request. With the paging of the KV cache and the introduction of the block-table, the attention operator now requires special handling to identify where in the GPU pool the relevant KV slots are located at, and then fetching them. To do this, the vLLM authors wrote a custom CUDA kernel implementing the attention operator with this dynamic fetching of KV slots via the block-table.

If you think of memory in blocks of size `B` KV slots, then consider what happens over the lifetime of a request. During prefill, the KV vectors for the prompt tokens are written into `ceil(prompt_length / B)` blocks, allocated as needed. All of these blocks are full except possibly the last one. During decode, each new output token's KV vectors go into the next free slot of that last block, and only once it is full do we allocate a fresh block for the next `B` KV slots.
This means that at any point in time, only the _last_ block of a request can be partially empty. Even if the request emits `<EOS>` right after using 1 KV slot in a new block, the worst-case total wasted allocation is at most `(B - 1)` KV slots, regardless of how long the prompt or output is. If you take our Llama-3-70B FP16 running example and assume B = 16 (vLLM default), this is a _maximum_ per-request wastage of `(16 - 1) x 328 KB (KV slot size) = 4.92 MB` - this is vastly smaller than the earlier wastage number of `0.76 x 2.68 GB ~= 2036 MB`. 400+ times smaller.
The other benefit is that now, because the allocator is allocating in these smaller and fixed block sizes and any block is usable via the intermediary block-table, a freed up block by some request `P` can be used by any other request `Q` for its token generation needs. Freed up memory blocks are essentially exchangeable between different requests _irrespective of the prompt size or #output tokens of these requests_.

Let's revisit our example from the previous section. Assuming the same h/w setup of 2xA100s with a combined 160GB of HBM2 memory, we discussed how loading a Llama-3-70B FP16 model would eventually yield ~10GB of free GPU HBM.
With standard attention, we could run at most 3 concurrent requests on such a set up, irrespective of how many tokens each request generated. With PagedAttention, assuming a 500-token prompt generating 1,500 output tokens consistently, we would only need `2000 (num tokens) x 328 KB (KV slot size) ~= 660 MB` space per-request. This means we would be able to run floor(10/0.66) = 15 concurrent requests. This is a massive **5x** increase in concurrency by switching to a paging model for managing KV cache memory.

PagedAttention is central to the concept of prefix caching, because it breaks down KV cache memory into fixed sized blocks and makes them interchangeably reusable via the block table. This allows for sharing cross-request common prefixes by having them share physical block (i.e. GPU pool index) references. In the next section, we'll investigate the use-case for prefix caching, and how it is implemented in vLLM.

## Resurfacing: Prefix Caching

In industry, it is very common for prompts to contain a 'system prompt' preamble/portion which describes the task at hand to the LLM, provides it with some examples if required, etc. to set up the problem and expected outputs. This system prompt is usually common to all requests made from that system for that use-case. Automated prefix caching would massively help speed up such requests.

The core idea of automatic prefix caching is simple. Because each token causally pays attention to only the previous tokens in decoder-only models like in our Llama example above, if 2 different requests share a prefix string, then we can look at their longest common prefix, and conclude that the KV vectors for tokens in that common prefix must be the exact same for both those requests. The KV vectors start differing for the 2 requests at the first position where generated tokens _differ for the very first time_, which by definition, is the immediate next token after where the longest common prefix ends.

So then a simple algorithm for prefix caching utilizing a hash table, is as follows:

1. Split the input request (assuming it was tokenized) into a sequence of block-sized (`B`) chunks of tokens.
1. For each _full_ **[1]** block of tokens, hash the concatenation of the previous block's hash, and the list of these token ids _keeping the ordering of the tokens the same_. I.e. `hash(prev_hash_id, list[token_id])` forms the hash key. The value associated with this hash key is the "physical" block id (i.e. index in GPU memory pool) of the KV slots for this block of tokens.
1. If there is a cache hit, this tells us 2 things: the previous block exists (in terms of the actual tokens), and it too was followed by the current _full_ block of tokens in the exact same order. I.e., we have seen this exact prefix before.
    - Upon a cache hit, we take the mapped physical block id and insert it in the request's block-table against the logical block id of this most recent block of tokens.
    - We increment a `ref_count` counter associated with this physical block id, to keep track of the number of live requests pointing to it.
    - No KV vectors were computed for this block of `B` tokens, and this is where the real power of prefix caching lies **[2]**. This leads to a direct speedup during the prefill phase and also reuse GPU memory, so no new memory was allocated as well.
1. If there is a cache miss, we allocate a free block (see description of this process below), set its `ref_count = 1`, add a mapping between the logical block id to physical block id in the request's block-table, and compute the KV vectors for these `B` tokens and write them into that block. Additionally, we insert a mapping `hash(prev_hash_id, block_token_ids) -> physical_block_id` into the prefix cache's hash table for potential use with future requests.

**[1]** We can only cache and re-use full blocks of `B` KV slots because partially full blocks may be actively in use by its owning-request. If we also assign this partially filled block to a second request's tokens, and the 2 requests have diverging generations at some point within that same block, the requests would race and overwrite each other's KV slots. If a block is full, however, it becomes immutable - it is frozen for the requests pointing to it, and only cleared out once it has been marked as having 0 references and another request needs it.

**[2]** Consider just the causal attention part of each Transformer layer in the Llama-3-70B model with a prompt of 500 tokens _without_ prefix caching. For each token $t_i$ we will compute $q_{i, l}$, $k_{i, l}$, and $v_{i, l}$ at layer $l \in \{1, \dots, 80\}$. These are computed using the token $t_i$'s hidden representation and the matrices $W_{Q_{l}}$, $W_{K_{l}}$, and $W_{V_{l}}$. Let's count the number of operations to compute only the K and V vectors for all tokens for each layer $l$. Each layer $l$ starts with some input $h_{i, l - 1}$ (of size `d = 8192`) for token $i$, and projects it into key and value vectors of size $d_{head} = 128$ across all 8 KV heads. This costs `500 (total tokens) x 8 (KV heads) x 2 (K and V) x 8192 (embedding dim) x 2 (multiply-add) x 128 (head dim) = 16.78 billion` operations per layer.

Let's say we now enable the KV prefix cache and find a match for the first 400 tokens. We now have to compute the KV vectors for just 100 tokens labeled `401, ..., 500`. This instantly reduces the total operations to 1/5th (`3.36 billion` operations—an 80% reduction). Also note: the above shows the calculations saved for just the K and V vector projections, but in reality, we can skip the _entire_ forward pass (query projections, attention outputs, and FFN layers) for the first 400 tokens. This is because for each cached token $t_i$, we do not need to compute the inter-layer hidden state $h_{i, l}$ for layer $l$ -- this was already done by a previous request, and its only outputs needed by future tokens, the vectors $k_{i, l}$ and $v_{i, l}$, were cached across all layers. In actuality, we therefore save a much larger magnitude of operations.

The goal is to assign each block in the request to a block id by the end of this process, by maximally re-assigning the prefix tokens to pre-existing blocks if this same prefix was seen before. Note how prefix caching's context is not per-request, but rather across-request - since this is where the prefix block reusability comes into play.

It is also worth noting, that while the use of prefix caching is most apparent in the prefill phase, we can also perform it during the decode phase. In the decode phase, once the next token's KV vectors have been determined at all layers, these vectors will be cached into KV slots in the currently open physical block for this request. If this next token causes this physical block to fill up, we add it to the prefix cache. This is especially useful in **multi-turn chat** use-cases. In this use-case, every time the chatbot answers, we add to the context of the entire history of the chat, and future requests can benefit from caching the KV vectors for this history.

For example, consider this interaction:

1. **Turn 1 User asks:** "How is the weather today in Sacramento, CA?" -> [System Prompt + Turn 1 User Prompt]
1. **Turn 1 Bot answers:** "It is sunny and not humid, with temperatures peaking at 96 degrees Fahrenheit." -> [System Prompt + Turn 1 User Prompt + Turn 1 Chatbot Output]
1. **Turn 2 User asks:** "Suggest some hikes nearby."

In the 2nd turn of this chat, consider the prefix that the chat system will generate. It will generate `[System Prompt + Turn 1 User Prompt + Turn 1 Chatbot Output + Turn 2 User Prompt]`. But notice that the `[System Prompt + Turn 1 User Prompt + Turn 1 Chatbot Output]` portion of this text is just the output that was generated by the LLM in the 1st turn! So if the 1st turn caches it's sequence of prefix blocks, the 2nd turn can use these cached KV vector references, and will only need to compute attention on the `Turn 2 User Prompt` portion for generating the next new token. This will dramatically reduce Time To First Token as well as meet multi-tenant throughput constraints too.

The algorithm needs to be a little meticulous in tracking the usage of blocks, freeing up blocks once a request indicates that it has completed processing, and reclaiming previously freed blocks.
In the event of a cache miss, a new free block must be sought for use with the current `B` tokens in question. There are 2 sources of free blocks:

- Blocks that have never been touched/used before, i.e. completely unused blocks, and
- Blocks that have been used before, but their parent requests have since finished and marked these blocks as eligible for reuse.

The first case is easy - we simply keep a list of physical block ids in memory, and keep popping off new unused ids from this list until this list is empty. Once this list is empty, we no longer have any more "untouched" physical blocks left for the remainder of the server's lifecycle, and so this "free list" becomes defunct and all sourcing of free blocks shifts to the second case.

In the second case, blocks that are no longer in use by _any_ of their parent requests are marked as 'inactive'. The fact that they are no longer in use is tracked by a per-block level `ref_count` variable. This counts the number of **active** requests referencing the block in some way. Every time a request frees its blocks, for each block freed, we _decrement_ its `ref_count`. We only mark a block inactive if its `ref_count == 0`. These 'inactive' blocks are stored in a Least Recently Used (LRU) queue, based on their access time. Note that we do _not_ clear out the contents of these blocks just yet. It is possible that a new request has some `B` tokens that match this block's history as well as contents, in which case we can simply rescue this block from the "inactive queue" and bring it back into active use with 0 additional work (as it already holds the KV vectors we desire). For sourcing a free block from the inactive queue in the case there's a complete cache miss, we pick off the block which has spent the longest time in this LRU queue. We also remove the corresponding entry for this block from the prefix cache hash table, since its contents are about to be overwritten completely.

## Prefix Caching In Action: Code Example

Here is Prefix Caching in action, written in Python. It assumes the existence of a `PhysicalBlock` class that stores important block-level details like the block id, content hash it serves, the reference count, and last access time.

```python
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict
import time
import hashlib

class PhysicalBlock:
    def __init__(self, block_id: int):
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.content_hash: Optional[int] = None
        self.last_accessed: float = 0.0

class PrefixCachingBlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        
        # 1. Physical blocks available
        self.blocks: List[PhysicalBlock] = [PhysicalBlock(i) for i in range(num_blocks)]
        
        # 2. Never-used / clean free blocks
        self.free_block_ids: List[int] = list(range(num_blocks))
        
        # 3. Lookup table: content_hash -> block_id
        self.cached_blocks: Dict[int, int] = {}
        
        # 4. LRU Eviction Queue for blocks with ref_count == 0:
        #    An OrderedDict[block_id, None] works great as an O(1) LRU queue.
        self.eviction_queue: OrderedDict[int, None] = OrderedDict()

    def _compute_hash(self, token_ids: Tuple[int, ...], prev_block_hash: Optional[int]) -> int:
        hasher = hashlib.sha256()
        # 1. Stream the integers into the hasher as fixed-width bytes
        # (8 bytes handles standard signed 64-bit integers)
        if prev_block_hash is not None:
           hasher.update(prev_block_hash.to_bytes(32, byteorder='big', signed=False))
        for num in token_ids:
            hasher.update(num.to_bytes(8, byteorder='big', signed=False))
            
        # 2. Extract the hex string and cast it straight to a base-16 integer
        return int(hasher.hexdigest(), 16)

    def allocate_sequence(self, token_ids: List[int]) -> Tuple[List[int], int]:
        n = len(token_ids)
        token_chunks = []
        for i in range(0, n, self.block_size):
           j = i + self.block_size
           token_chunks.append(token_ids[i:j])
        if token_chunks and len(token_chunks[-1]) != self.block_size:
           # Ignore last chunk of tokens if not block-aligned.
           token_chunks = token_chunks[:-1]

        prev_hash = None
        allocated_block_ids = []
        num_hits = 0
        for i in range(len(token_chunks)):
          block_hash = self._compute_hash(token_chunks[i], prev_hash)
          if block_hash in self.cached_blocks:
            block_id = self.cached_blocks[block_hash]
            if block_id in self.eviction_queue:
               del self.eviction_queue[block_id]
            num_hits += 1
          else:
            # Cache miss. We have to find a free block and alloc it.
            try:
              block_id = self._get_free_block()
            except RuntimeError as re:
              print(f"Error message: {re}; reverting sequence caching.")
              self.free_sequence(allocated_block_ids)
              return ([], 0)

          # Reset key fields in the block and increment refs.
          pblock = self.blocks[block_id]
          pblock.content_hash = block_hash  # Safe operation.
          pblock.ref_count += 1
          pblock.last_accessed = time.time()
          self.cached_blocks[block_hash] = block_id

          allocated_block_ids.append(block_id)
          prev_hash = block_hash

        return (allocated_block_ids, num_hits * self.block_size)
           

    def free_sequence(self, block_ids: List[int]) -> None:
        for bid in block_ids:
           pblock = self.blocks[bid]
           pblock.ref_count -= 1
           pblock.last_accessed = time.time()
           if pblock.ref_count == 0:
              self.eviction_queue[bid] = None

    def _get_free_block(self) -> int:
        if self.free_block_ids:
          return self.free_block_ids.pop()
        elif self.eviction_queue:
          result, _ = self.eviction_queue.popitem(last=False)
          pblock = self.blocks[result]

          # Clean up cached_blocks; remove the old hash reference.
          # This same block will now be inserted into the dict as
          # a new value, associated with a _potentially different_
          # content hash.
          if pblock.content_hash:
            del self.cached_blocks[pblock.content_hash]

          # Reset the block; we will reference it when we see cache-hits.
          pblock.ref_count = 0
          pblock.content_hash = None
          pblock.last_accessed = time.time()
          return result
        else:
           raise RuntimeError("Out of Memory")


# Quick Test Harness
allocator = PrefixCachingBlockAllocator(num_blocks=4, block_size=4)

print("--- Query 1: Cold start ---")
prompt_1 = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 chunks
blocks_1, hits_1 = allocator.allocate_sequence(prompt_1)
print(f"Allocated blocks: {blocks_1}, Cached tokens hit: {hits_1}")
# Expect: blocks_1 = [3, 2] (or [0, 1]), hits_1 = 0

print("\n--- Query 1 completes, freed ---")
allocator.free_sequence(blocks_1)
print(f"Eviction queue: {list(allocator.eviction_queue.keys())}")

print("\n--- Query 2: Shared prefix + new suffix ---")
prompt_2 = [1, 2, 3, 4, 99, 99, 99, 99]  # Chunk 0 matches!
blocks_2, hits_2 = allocator.allocate_sequence(prompt_2)
print(f"Allocated blocks: {blocks_2}, Cached tokens hit: {hits_2}")
# Expect: Chunk 0 reuses Query 1's block! hits_2 = 4

print("\n--- Query 3: Force OOM and Eviction ---")
prompt_3 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]  # Needs 3 blocks
blocks_3, hits_3 = allocator.allocate_sequence(prompt_3)
print(f"Allocated blocks: {blocks_3}, Cached tokens hit: {hits_3}")
```