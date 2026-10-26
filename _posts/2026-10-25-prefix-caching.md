---
layout: post
author: Nishant Kelkar
title: Prefix Caching for KV Caches
tags: machine-learning deep-learning system-design
---

Suppose you have a request whose prompt and generated output together span 8,192 tokens. The embedding space of these tokens is `d` dimensional, giving us an `8192 x d` matrix of token embeddings. During both the prefill and decode stages, we would compute many thousands of key and value vectors (KV vectors) across these 8,192 tokens. This blog post goes deeper into how vLLM manages memory for these KV vectors (the so-called "KV Cache") and an optimization that kicks in when prompt prefixes are shared across requests (which in real-world scenarios is always true, e.g. system prompts).

* This will become a table of contents (this text will be scrapped).
{:toc}

## The Problem

Imagine you have a request spanning 8,192 tokens (prompt + output). Once you project each of these tokens into the $d_{head}$ dimensional space (using the $W_Q$, $W_K$, and $W_V$ projection matrices), each token's embedding needs to "attend to" all the tokens that came before it. Specifically, each token $t_i$'s query vector $q_i \in \mathbb{R}^{1 \times d_{head}}$ multiplies ${K}^T_{j \le i}$ of dimension $d_{head} \times i$.

How many values must we store for token $t_i$ so future tokens can attend to it?
* We need **1 key vector $k_i$** and **1 value vector $v_i$** (both of shape $1 \times d_{head}$), since future tokens $t_{i+1}, t_{i+2}, \dots$ reuse them to compute attention over $t_i$.
* We **do not** store the query vector $q_i$, because future tokens $t_{i+1}, t_{i+2}, \dots$ never use $q_i$.

Thus, for each token we store $(2 \times d_{head} \times H \times L)$ floating-point values, where $H$ is the number of KV heads per Transformer layer and $L$ is the number of layers in the model. We'll call the memory needed to hold one token's key and value vectors across every layer and KV head a **KV slot** (or just **slot**).

To make this concrete, consider **Llama-3-70B** in FP16 precision ($L = 80$ layers, $H = 8$ KV heads, $d_{head} = 128$, and 2 bytes per FP16 value):

$$\text{1 KV slot} = 2 \times 128 \times 8 \times 80 \times 2 \text{ bytes} = 163,840 \times 2 \text{ bytes} \approx 328 \text{ KB}$$

For a single request filling Llama-3-70B's 8,192-token context window (prompt + output), the KV cache alone requires:

$$8,192 \text{ slots} \times 328 \text{ KB} \approx \mathbf{2.68 \text{ GB}}$$

In the grand scheme of things, 8,192 tokens isn't even a long context by modern standards (where 1M-token contexts are commonplace).

### Decode Is Memory-Bandwidth-Bound

Now consider the decode stage. To generate each new token, the model attends over every previous token, which means reading the _entire_ KV cache out of GPU memory (HBM): all the prompt tokens' KV vectors plus every output token generated so far. This cache grows by ~328 KB per token, reaching the full 2.68 GB by the end of our 8,192-token request.

Suppose we serve this model on **2xA100 GPUs (80 GB each)**, giving us ~4 TB/s of combined HBM bandwidth. Reading 2.68 GB at 4 TB/s takes:

$$\frac{2.68 \text{ GB}}{4000 \text{ GB/s}} \approx \mathbf{0.7 \text{ ms per output token}}$$

That is up to ~0.7 ms per token spent _just moving KV vectors_, before any math happens. And the math itself is cheap: each 256-byte key vector ($128 \times 2$ bytes) takes part in a single 128-dim dot product with the query (~256 FLOPs, or **~1 FLOP per byte read**). An A100 needs **~150 FLOPs per byte** (312 TFLOPS / 2 TB/s) before its compute units become the bottleneck. Attention during decode is therefore heavily **memory-bandwidth-bound**: the GPU spends most of its time waiting on memory, not computing.

On its own, 0.7 ms looks small next to the ~35 ms it takes to stream the model's 140 GB of weights each step. But **model weights are read once per step for the whole batch**, while **every request must read its own KV cache**. Batching many requests together is how we amortize the cost of reading the weights—which raises the real question: _how many requests can we even fit in memory?_

Our two A100s have 160 GB of HBM2 in total:
* **Model weights:** `70B x 2 bytes = 140 GB`
* **CUDA runtime & activations:** `~10 GB`
* **Remaining for KV cache:** **`~10 GB`**

At 2.68 GB per request, we can hold at most $\lfloor 10 / 2.68 \rfloor = \mathbf{3 \text{ concurrent requests}}$—and even that is a _best-case_ estimate assuming zero memory fragmentation.

---

## Solving A Different Problem: Fragmented Memories

When a request arrives, its `prompt_length` is known immediately, but its output length is unknown _a priori_ and varies from request to request. Before vLLM's [PagedAttention paper](https://arxiv.org/pdf/2309.06180), serving systems allocated a contiguous chunk of `prompt_length + max_tokens` KV slots upfront for every request (capped at `max_seq_length`, and defaulting to `max_seq_length` whenever `max_tokens` was left unset). `max_tokens` here is the client-supplied indicator for the maximum number of output tokens it wants to generate.

Reserving `prompt_length + max_tokens` contiguous KV slots upfront leads to **3 types of memory waste**:

1. **Reserved slots (opportunity cost):** The KV slot for the 100th future token is held from the moment the request begins generating its 1st token. Even if that 100th token is eventually generated, that memory sits idle in the meantime instead of serving other concurrent requests.
2. **Internal fragmentation (over-allocation):** A request usually hits an `<EOS>` token and stops at `T` total tokens (prompt + output) where `T < prompt_length + max_tokens`, wasting `(prompt_length + max_tokens - T)` allocated KV slots.
3. **External fragmentation:** Because contiguous reservations vary in size across requests, freeing a finished request leaves a contiguous hole of `G` KV slots in HBM that cannot serve any new request needing `> G` slots—stranding free memory in unusable gaps.

```
Pre-PagedAttention (Contiguous Reservation per Request, max_seq_length = 8192):
+------------------+-------+----------------------------------+-----+------------------+
| Prompt (500)     |Gen so | Reserved for future / Unused     | Gap | Request B ...    |
| [██████████████] |far[██]| [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] | (G) | [████████░░░░░░] |
+------------------+-------+----------------------------------+-----+------------------+
                           |<---- ~76% wasted (2,036 MB) ---->| ⬆
                                                                ⬆-- External frag (< Req C)
```

To see the impact on Llama-3-70B (`max_seq_length = 8192`, `max_tokens` unset): each request reserves `8192 x 328 KB = 2.68 GB` upfront. If a request has a 500-token prompt and generates 1,500 output tokens (2,000 tokens total), it uses only **~655 MB** of its 2.68 GB reservation—**wasting ~76% (~2,036 MB) of its allocated KV slots**.

### How PagedAttention Fixes This

PagedAttention decouples upfront reservation from physical allocation by taking a page (pun intended) out of OS virtual memory design. It partitions the GPU's KV memory pool into fixed-size **blocks** of `B` KV slots (`B x 328 KB` in our example). In vLLM's experiments, **$B = 16$ KV slots** hit the sweet spot between blocks being too coarse (bringing back internal fragmentation) and too fine (hurting GPU memory-access coalescing and parallelism inside the attention kernel).

Under the hood, each request gets a private **block-table** that maps **logical block IDs** (`0, 1, 2, ...`) to **physical block IDs** (indices into the global GPU memory pool allocated at startup):
* Token $i$'s KV vectors go into logical block `i // B` at slot offset `i % B`, which the block-table translates into the actual physical block index in HBM.
* Standard attention kernels expect contiguous KV tensors per request, so vLLM uses a custom CUDA kernel that reads the block-table on the fly to gather non-contiguous physical blocks during attention.

```
PagedAttention (Block Size B = 16 KV slots):
Logical Blocks (Request P)        Block Table        Physical Block Pool (GPU HBM)
+------------------+              +----+----+        +----------------------------+
|LB 0: [██████████]| (16/16 full) | 0  | 7  | -----> |PB 7: [██████████] (Req P)  |
|LB 1: [██████████]| (16/16 full) | 1  | 2  | --+    |PB 1: [██████████] (Req Q)  |
|LB 2: [███░░░░░░░]| (3/16 used)  | 2  | 5  | -+|--> |PB 2: [██████████] (Req P)  |
+------------------+              +----+----+  |     |PB 3: [░░░░░░░░░░] (Free)   |
         ^                                     |+--> |PB 5: [███░░░░░░░] (Req P)  |
 Only the LAST block can have empty slots!           +----------------------------+
```

Tracing a request over its lifetime shows how all three wastes disappear:
* **On-demand allocation (fixes #1):** During prefill, prompt tokens fill $\lceil \text{prompt\_length} / B \rceil$ blocks allocated on demand. During decode, each new output token's KV vectors go into the next free slot of the last block; a fresh block is pulled from the pool's free list only when that block fills up.
* **Bounded internal fragmentation (fixes #2):** At any time, **only the last block of a request can be partially empty**, bounding waste to at most $B - 1$ KV slots regardless of prompt or output length. For Llama-3-70B with $B = 16$, worst-case waste per request is $(16 - 1) \times 328\text{ KB} = \mathbf{4.92\text{ MB}}$—over **400x smaller** than the ~2,036 MB wasted before.
* **Zero external fragmentation (fixes #3):** Because all blocks have the same fixed size $B$ and are addressed indirectly through the block-table, any block freed by request `P` can immediately be reused by any other request `Q`, irrespective of prompt or output lengths.

**The Concurrency Payoff:** Revisiting our 2xA100 setup with ~10 GB of free KV memory, our 2,000-token request (500 prompt + 1,500 output) now consumes only `2000 x 328 KB ~= 660 MB`. Instead of 3 concurrent requests, we can now run $\lfloor 10 / 0.66 \rfloor = \mathbf{15 \text{ concurrent requests}}$—a **5x increase in concurrency**!

PagedAttention also unlocks **prefix caching**: because KV memory is broken into fixed-size blocks addressed via a lookup table, multiple requests sharing a common prompt prefix can simply point their block-tables at the _exact same physical blocks_.

---

## Resurfacing: Prefix Caching

In production LLM applications, requests almost always share a long **system prompt** (instructions, tool definitions, few-shot examples) before the user's query. Because decoder-only models like Llama use causal attention—where each token only attends to earlier tokens—if two requests share a prefix of tokens, **the KV vectors for every token in that shared prefix are 100% identical**. Their KV vectors only begin to diverge at the first token after the longest common prefix.

<figure class="blog-fig">
  <img src="/assets/images/prefix-cache-block-sharing.png">
  <figcaption>Figure 1. Requests A & B share a 32-token system prompt (B=16)</figcaption>
</figure>

### The Hash-Table Algorithm

We can share prefix blocks across requests using a global hash table (`cached_blocks`):

1. **Chunk into blocks:** Split the tokenized request into blocks of `B` tokens.
2. **Chained hashing (full blocks only [1]):** For each _full_ block of `B` tokens, compute `block_hash = hash(prev_block_hash, block_token_ids)`, preserving token order. This hash key uniquely identifies the entire token sequence from the start of the prompt up to the end of this block, and maps to a physical block ID in the GPU pool.
3. **On a Cache Hit:** Both this block and its entire preceding prefix match a previously computed block in GPU memory:
    - Map the request's current logical block ID to this existing physical block ID in its block-table.
    - Increment the physical block's `ref_count` (tracking how many active requests reference it).
    - **Skip computing KV vectors for these `B` tokens entirely [2]**, saving both prefill compute and GPU memory.
4. **On a Cache Miss:** Allocate a free physical block (process detailed below), set its `ref_count = 1`, map the logical block ID to it, compute and write the `B` tokens' KV vectors into it, and insert `block_hash -> physical_block_id` into `cached_blocks`.

> **[1] Why only cache _full_ blocks?** A partially filled block is still actively being written to as its owning request generates new tokens. If two diverging requests shared a partially filled block, they would race and overwrite each other's KV slots. Once a block is completely full (`B` slots), it becomes **immutable**—safe for any number of requests to share read-only until its `ref_count` drops to 0 and it is reclaimed.

> **[2] Just how much prefill compute does a cache hit save?** Consider computing the K and V projections in one layer of Llama-3-70B ($d_{\text{model}} = 8192$, $H = 8$ KV heads, $d_{\text{head}} = 128$) for a 500-token prompt _without_ prefix caching:
>
> $$500 \text{ (tokens)} \times 8 \text{ (KV heads)} \times 2 \text{ (K, V)} \times 8192 \text{ ($d_{\text{model}}$)} \times 128 \text{ ($d_{\text{head}}$)} \times 2 \text{ (multiply-add)} = \mathbf{16.78 \text{ billion FLOPs/layer}}$$
>
> If the first 400 tokens hit in the prefix cache, we only compute KV projections for the remaining 100 tokens (`401..500`), dropping the cost to **1/5th (`3.36 billion` FLOPs/layer—an 80% reduction)**. Better yet, we don't just save the K and V projections: we **skip the entire forward pass** (query projections, attention outputs, and FFN layers) for those 400 cached tokens! Because an earlier request already computed the inter-layer hidden states $h_{i, l}$ and cached their only outputs needed by future tokens ($k_{i, l}$ and $v_{i, l}$ across all layers $l \in \{1, \dots, 80\}$), those 400 tokens aren't fed into the model at all.

### Caching Decode Tokens: Multi-Turn Chat

While prefix caching is easiest to picture during prefill, it also happens during **decode**: as each generated token's KV vectors are written into the request's currently open physical block, **the moment that block fills up (`B` tokens), it is hashed and inserted into the prefix cache**.

This shines in **multi-turn chat**:
1. **Turn 1 User asks:** `"How is the weather today in Sacramento, CA?"` $\rightarrow$ `[System Prompt + Turn 1 User Prompt]`
2. **Turn 1 Bot answers:** `"It is sunny and not humid, peaking at 96°F."` $\rightarrow$ `[System Prompt + Turn 1 User Prompt + Turn 1 Bot Output]`
3. **Turn 2 User asks:** `"Suggest some hikes nearby."` $\rightarrow$ `[System Prompt + Turn 1 User Prompt + Turn 1 Bot Output + Turn 2 User Prompt]`

When Turn 2 arrives, its prefix `[System Prompt + Turn 1 User Prompt + Turn 1 Bot Output]` was already cached block-by-block during Turn 1's prefill and decode! Turn 2 only needs to run prefill on `Turn 2 User Prompt`, slashing Time-To-First-Token (TTFT).

### Block Lifecycle: The Free List & Inactive LRU Queue

When a cache miss occurs, the allocator needs a free physical block. Free blocks come from two sources, in priority order:

<figure class="blog-fig">
  <img src="/assets/images/prefix-cache-state-machine.png">
  <figcaption>Figure 2. State transitions of a physical block in Prefix Cache</figcaption>
</figure>

1. **Untouched Free List:** At startup, all physical block IDs sit in a simple free list. We pop unused IDs from this list until every block in the GPU pool has been used at least once, after which this list is permanently empty.
2. **Inactive LRU Queue (`eviction_queue`):** When a request finishes, we decrement `ref_count` on each of its blocks. If a block's `ref_count` reaches `0`, we mark it **inactive** and push it into an **LRU eviction queue**—_without_ wiping its KV vectors or removing its hash from `cached_blocks`:
   * **Rescue on Hit:** If a new request gets a cache hit on an inactive block, we remove it from the LRU queue, increment `ref_count = 1`, and reuse its intact KV vectors with zero compute.
   * **Evict on Miss:** If we need a fresh block on a cache miss (and the untouched free list is empty), we pop the **least-recently-used** block from the front of the LRU queue, delete its old hash from `cached_blocks`, and overwrite it with the new block's KV vectors.

### Summary: Putting the Numbers Together

Across our running **Llama-3-70B FP16** example on **2xA100 (10 GB KV cache budget)** for requests with a **500-token prompt (400-token shared system prompt + 100-token user query) and 1,500-token output**:

| Memory Management Strategy | KV Memory per Request | Max Concurrent Requests (in 10 GB) | Max Wasted KV Memory / Req | Prefill Tokens Computed (Requests 2..N) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Contiguous (`max_seq_len = 8192`)** | `2.68 GB` | **3** | `~2,036 MB` (76%) | `500 tokens` (100%) |
| **2. PagedAttention (`B = 16`)** | `~660 MB` | **15** (5x) | `4.92 MB` (< 1%) | `500 tokens` (100%) |
| **3. PagedAttention + Prefix Caching** | `~525 MB` unique + `131 MB` shared once | **18** (6x) | `4.92 MB` (< 1%) | **`100 tokens` (80% saved!)** |

---

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
          if pblock.content_hash is not None:
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