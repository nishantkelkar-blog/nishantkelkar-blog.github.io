---
layout: post
author: Nishant Kelkar
title: Design a Chat System
tags: system-design
---

This is a pretty common systems design question.
The first thing you have to remember, is that it is impossible to design a complete chat system in 45 - 60 minutes.
So give up on that hope at the beginning.
The second thing to remember, is figuring out how your interviewer wants to structure the interview.
Remember, you are driving the design, but they are calling the shots on _what_ to design.
So while you may want to collect requirements and then jump into a high-level overview to get buy-in, the interviewer may want you to just focus on the data model.
Pay close attention to your interviewer's expressions, body language, and verbal cues. These are EXTREMELY important.

## Requirements gathering

Here are some requirements that you may end up gathering from the interviewer:

```text
1. Support 10M daily active users (DAU).
2. Support 1:1 chat, as well as 1:many group chats.
3. Group size is limited to 100 users/group.
4. Message sizes are limited to 10KB/message.
5. Each user on average sends 100 messages.
6. Message is limited to text modality only.
7. Data retention: forever.
```

How do you know what questions to ask and what requirements are relevant to the design? Practice :)

Also note that we are **_not_** building the system for:

```text
1. Multi-modal messages, with audio/video/image components.
2. Large group chats.
3. Last-seen feature.
4. Giphy/other integrations.
5. Message threads.
6. Push notifications.
```

It is important to think about some chat systems that you know of, when thinking about what _not_ to include in the design (e.g. WhatsApp, Slack, ...).

You need to point out some key issues right off the bat.
This lets your interviewer know that some important things have already caught your eye.
Particularly, there are 2 challenges that stand out immediately:

- How will user-to-user communication work? Especially, once a sender sends a message to the chat system's servers, how will that message be delivered to the recipient?
- How will we store the message data? And how much storage will we need? Our TTL here is essentially $\infty$.

We will see how to tackle these 2 problems in the sections below.

## High-level design

At a high level, we have the following interaction happening:

<figure class="blog-fig">
  <img src="/assets/images/chat-system-high-level.jpg">
  <figcaption>Figure 1. Super high level requirements from the system</figcaption>
</figure>

In a client-server setup, usually we think of the client initiating the request and server responding.
In our design, this works fine on the sender's side, as they initiate a simple HTTP connection and send the message data along with recipient information to the system.
However, how will the recipient receive the message?
We want the server to "push" the message data coming from user `E` to the client (i.e. user `F`).

Regular HTTP does not allow for this, but you can work around it via strategies like polling, or long-polling.
These are wasteful in compute though, as polling requires repeatedly establishing a TCP handshake connection and long-polling also has particularly inactive clients connecting periodically to the servers.

What we need here are **WebSockets** (WS).
WebSockets are a relatively new technology (~2008), to deal with requirements around long-lived connections.
A WebSockets connection piggy-backs over a regular HTTP connection, but then is 'upgraded' to a WS connection upon request from the client (or server).
The initial HTTP request literally has the `Connection: Upgrade` and `Upgrade: websocket` header, which upgrades the connection to be a WS bidirectional one.
For a deep-dive into WebSockets, their history, and code examples, check out this [Ably](https://ably.com/topic/websockets) article.

So we now know that recipients will need a WS connection with a chat server that 'pushes' message data to it whenever it is connected.
Let's recap then: user `E` sends a message intended for `F`, which first goes to a load balancer.
The load balancer forwards that request to one among `N` stateless servers, which perform authentication checks, request validation, etc.
Once these checks are performed and pass, the request is sent to a conceptual chat routing system, which figures out what the chat server is for `F`, and forwards the message there.
This routing system also collects an ack from the chat server, which can then also be bubbled-up to `E` as a 'Delivered' notification.

The chat server has 2 functions: first, it must persist the message, in-case immediate delivery fails. Second, if `F` is connected and online, it must use the WS connection to send the client device of `F` the message from `E`.
In the event that `F` is offline, the chat server also needs to remember the order of messages (important from a group messaging perspective) so that it may deliver them in this order once `F` is back online.
This slightly more detailed view of the system is shown below in fig. 2.

<figure class="blog-fig">
  <img src="/assets/images/chat-system-digging-deeper.png">
  <figcaption>Figure 2. Going deeper. Here we show the protocols used for send/receive, as well as initial request processing</figcaption>
</figure>

At this point, you have to work with your interviewer to get **buy-in**.
They have to say OK to your initial proposal, for you to proceed to the deep dive.
If they have any suggestions (in the form of questions), be diligent to go over your high-level design and make modifications if necessary.
It is not a sign of weakness if you change your initial design based on the interviewer's feedback!
If anything, it shows that you are willing to make changes in your work based on others's inputs and requirements.
This is always a desired trait in a future-colleague.

## Deep dive

Once the high-level design is agreed upon, we now are left with answering the following 3 questions:

1. What is the data model here, and what data storage to use?

2. How does the chat routing system know what chat server to reach out to?

3. How does the chat server queue incoming messages in order, and also persist these? And how does group messaging work?

Let's look at (1) first.
Based on our initial requirements, we have 10M DAU and 100 messages sent daily, each up to size 10KB.
Assuming each message is stored in the storage _at least once_, we have to store:

10M x 10KB x 100 ~= 10M x 1MB ~= **10 TB of data/day**.

Assuming a capacity planning horizon of 5 years, this generates **~18.25 PB of data in 5 years**.
We need a NoSQL solution here, as it is better at scaling horizontally, as well as good for unstructured text data like the one in our case.
We do not need to support search across past messages here, but otherwise we would need to also index this data in a search index like Solr.

A good choice here is a key/value store, where the primary key is a `(sender_id, recipient_id), message_id`, and the value is the `message_text`.
We can have the `(sender_id, recipient_id)` pair be the partition key, and the `message_id` be the clustering key so that within a given partition, all the messages are sorted by `message_id`.
A columnar store like Cassandra works well here, as we can add more nodes to scale it horizontally, and also have a random hashing function that distributes the `(sender_id, recipient_id)` equally among the nodes.
Because `message_id` is a clustering key here, we can support queries like "For `sender_id = 123` and `recipient_id = 456`, give me all messages with `message_id > 100`" very efficiently.

The primary entities in our setup are: Users, Groups, UserToUser, and UserToGroup.

Because we have O(~millions) of users, and user-user and user-group interactions are still fairly small (say, ~20 groups/person and ~100 individual chats/user), we are looking at about 10M x ~(100 + 20) = 1.2B rows.

The schema below shows an initial setup for storing basic information about users, groups, and their interactions.

```sql
-- 8 (id) + 100 (name) + (8x2) (location) + 13 (last_login_ts) + 1000 (dp_img_path) = 1137 bytes/row
User:
- id BIGINT (autogenerated),
- name VARCHAR(100),
- location JSON,
- last_login_ts TIMESTAMP,
- dp_img_path VARCHAR(1000),
...

-- 8 (id) + 100 (name) + (8x100) (admins) + 1000 (dp_img_path) = 1908 bytes/row
Group:
- id BIGINT (autogenerated),
- name VARCHAR(100),
- admins LIST<BIGINT>,
- dp_img_path VARCHAR(1000),
...

-- 8 (id) + 8 (userid1) + 8 (userid2) + 1 (status) = 25 bytes/row
-- 1:1
UserToUser:
- id BIGINT (autogenerated),
- userid1 BIGINT (fk User),
- userid2 BIGINT (fk User),
- status BOOLEAN,
...

-- 8 (id) + 8 (userid) + 8 (groupid) = 24 bytes/row
-- many:many
UserToGroup:
- id BIGINT (autogenerated),
- userid BIGINT (fk User),
- groupid BIGINT (fk Group),
...
```

From the above, we see we have to account for the following storage size:

(1137 x 10M) + (1908 x 20M) + (25 x 1000M) + (24 x 20M) ~= **70GB** of storage.

We can use a relational database here, as this is a perfectly manageable dataset size for a RDBMS.
Additionally, we can set up an index on the `groupid` field in the `UserToGroup` table so that we can speed up queries that look to find all the users that are part of a given `groupid` and avoid a table scan.

<figure class="blog-fig">
  <img src="/assets/images/chat-system-deep-dive.png">
  <figcaption>Figure 3. Deep dive into how chat server and routing works</figcaption>
</figure>

Questions (2) and (3) are answered in the diagram above.
The chat routing service is a stateless service that relies on another service, shown as the 'Connections Service' (CS) in fig. 3 below.
It fetches the chat server address from the CS, forwards the message to the chat server (possibly w/ retries), and bubbles up the `ACK` that it receives from the chat server.
Both the chat routing and CS can operate as REST services on top of HTTP(S), or even as gRPC services, as bidirectional communication isn't required here.

The CS provides server-discovery value.
For a given recipient `user_id`, it returns the `server_id` (and consequently, IP address) of the chat server which holds the WS connection with the recipient user.
An addition of a new chat server first registers the server with the CS.
The CS is also in charge of making sure that chat servers are alive by expecting periodic heartbeats from these, as well as establishing new connections whenever a user logs in/goes online.
For new connection requests, the CS could take into account details like present chat server loading, geographical location of user, device type, etc.
If a chat server suddenly goes down, the user's client device reinitiates a WS connection by first asking the CS what new chat server to establish a connection with.
Fig. 3 shows a small snippet of a possible DB schema that backs the CS - the main 2 tables are a server details table, and a mapping table from users to their respective chat servers.
If `last_alive_ts` for a particular server is too old (i.e. no heartbeat was received for a while), the chat server is marked dead, and new connection requests are not sent to it.

Fig. 3 also shows how a chat server operates.
Upon receiving a message, it is first logged to disk to an append-only replay log.
Only after this is an `ACK` sent back to the chat routing service.
This persistence is important to avoid loss of data in the event of a server crash.
Asynchronously, we also store the data into our messages datastore (e.g. Cassandra, per above discussion).
The chat server maintains a single queue for each user that it is connected with.
Upon receiving a message request, it unpacks it to figure out what user the message is intended for.
It then enqueues a tuple containing the message and the sender in the queue for that user (in this case, user `F`).
On the other end of the queues, a group of consumer threads pop messages of the queues and send them to the users over the WS held connection.

It's interesting to note, that the use-case for group messages can be extended with this above setup quite easily.
A group can be considered as a 'multi-person' user in our design, essentially.
The flow of events up until the message lands on the chat server is exactly the same for group messaging.
Upon popping the message off of this 'multi-person' user queue, we have `N` people to send this message to now, instead of just 1.
So there's no single WS connection we can send this message on, there's rather `N` of them.
In that case, we can simply query the CS for a list of actual user names for the given group "user id" as well as their corresponding chat servers, and then have the consumer thread forwards the message out to each of those users's chat servers.
Because each group is `< 100` users, this mechanism is fine, and can easily be handled by something like a dedicated group forwarding threadpool on the chat server.
This is also great from a persistence/logging perspective, since we need to log the message to disk only once (the forwarding API on the chat server can ignore persisting the message).
We could also put a cache next to chat servers that serve group "user"s, so that we do not have to query the CS every time.
Note that such a cache would need to be updated upon `user -> chat server` mapping updates within the CS.

## Further considerations

The design at this point is complete for the requirements we initially gathered.
It is by no means a complete product, however (remember -- we accepted that this was not going to happen anyways).
In an interview, it makes sense to quickly (and I mean, _really_ quickly -- in ~1 minute/point) discuss some additional points that may be relevant:

1. How does authentication work here?
2. How does chat server crash recovery work? Specifically, how may the logs be formatted so as to re-create the per-user queues?
3. How can we make the CS more available?
4. Is DB partitioning (sharding) possible here? What would you shard on?
5. Is it possible to re-use an existing WS connection with a chat server to also _send_ messages?

And that's a wrap!
