---
layout: post
author: Nishant Kelkar
title: Design an A/B Testing System
tags: system-design
---

Design an A/B testing and experimentation platform.
Users should be able to specify the environments to use for the A/B test, percentage traffic going to the A/B test, and how performance is to be computed for both the base and test sides.


## Requirements gathering

Having briefly worked on such a platform, I can confidently say that designing an A/B testing platform is a highly complex task.
Furthermore, to do it in a complete manner in 30-45 minutes is impossible.
Thus you should immediately narrow down the scope of the problem for the purposes of the interview.
Here are a few questions worth asking:

```text
1. To clarify, is the platform going to handle logs collection and metric computation? Or is that part of another system we rely on?
2. What is the A/B testing platform for? I.e., what are we comparing?
3. What are the top 2-3 features that we will be designing for in this interview?
4. What is the traffic that we expect to see? This is for both running new tests/checking on existing tests, as well as traffic from a service usage perspective.
5. What are the artifacts that we need to keep track of? I.e. what are the artifacts that the platform will push to production to initiate the test?
```

Let us assume the following answers to these questions:

```text
1. For simplicity, assume that logs collection and metrics computation is handled by a separate team, which exposes a REST API to us to collect results. We can ask this separate service to compute specific metrics for the comparison, via a POST API call, for example.
2. We are interested in testing 2 kinds of changes in an ML setting. Config changes (i.e. the model stays the same, but the configuration changes), and model changes (i.e. the entire model itself changes and we want to test this change).
3. Users of the platform should have access to the following information:
- A page that allows for creation of a new experiment, and associated setup.
- A page to search by experiment id/experiment creator name.
- A page that provides tabulated results for the metrics collected so far for both base and test sides.
4. We expect to run O(~100) experiments daily, and the service(s) that the experiments run on experience a traffic of ~10K QPS.
5. Based on the answer in #2, we need to track the configuration changes for each experiment, and/or the model version change.
```

## Data model and high-level design

From the answers given for our requirements, we can conclude 2 things:

1. The services for which we want to run the experiments have a high QPS / low latency. This means we cannot have downtime for initiating an experiment.
2. Our system need not concern itself with lower-level details about the service logic itself. We just care about pushing configuration or model changes to the service.

At the top-level, we have a collection of servers that together run the code for a specific service. This service needs to support 2 APIs in addition to the business-logic APIs:

```text
addExperimentConfiguration(Config config, ExperimentSettings settings)
addModelConfiguration(ModelConfig config, ExperimentSettings settings)
```

`addExperimentConfiguration` creates a new thread within each of the servers that holds a pointer to the control model, but replaces the control config with the `config` parameter value. `settings` provides the server information about the experiment, such as `experiment_id`, `alloc_pct` (allocation percentage, e.g. 0.5%, 1%, 5%, etc.).

`addModelConfiguration` also creates a new thread within each server. It however, reads in an experimental model from a model versioning store, the information of which is provided in the `config` configuration (e.g. model version, model store url, etc.). Like in the previous case, a `settings` object provides the server with information about the experiment.

Whenever the servers receive a request, we could run experiments in 2 ways. One, the request has information about what experiment the user is already slotted into, and so the server just routes the request to the associated `thread` that is running the experiment. In this method, each client will need to keep track of what experiment it is in (if any), and make sure to attach that information in each request to the load balancer of our service. This experiment information can easily be tracked via a browser-based cookie, or a device-stored configuration file (if using a smart phone, for example).

Alternatively, each request is randomly routed to one of the `threads` running an experiment, weighted by the `alloc_pct`. In this method, one advantage is that the client can be extremely lightweight, and does not need to keep track of experiment information. However, because the requests may be routed to _different_ experiments for each API call, this will lead to potentially inconsistent behavior for the end-user, which may result in a worse user experience. For example, suppose our service serves tiles for a maps application, and we have 2 experiments: one where based on time of day, we automatically switch to dark mode, and another where we are always in light-mode. Now, to our end-user, they will see the maps UI flip-flop between dark and light modes as they pan around the map/zoom-in or zoom-out (as this will make multiple API calls to our tiles service). This makes for a very bad user experience.

Given the need to maintain consistent user experience, we go with the first option of having the browser/device client maintain state of the experiment that it is in. This also means that periodically, the client will have to report user behavior (tagged with experiment information) to our logging service. This tuple of `(experiement, usage info)` is what will be used by logging & analytics to collect metrics about how effective the experiment is.

An A/B test pusher service makes calls to the above APIs when someone creates an experiment via the A/B platform UI. Some care must be taken here before the pusher initiates the experiment in production, as these changes directly affect user experience. We need to have some checks run before a config/model config is pushed out. Specifically, we need to check that:

1. The total allocation across all experiments + control = 100% always. There are alternatives. For example, you may allow a user to enroll in multiple tests. But this complicates matters and you have to be careful. For example, you may run into inter-test interference, where 2 tests running on the same user may influence each other's outcome.
2. The change proposal must be reviewed and approved by a service owner.
3. All unit and CI tests must pass with the proposed changes. If they don't, we cannot trust this change on production traffic.

Let us look at the key data components. We have the following entities:

```sql
Experiment:
- id            BIGINT (autogenerated),
- created_at    TIMESTAMP,
- status        ENUM,   -- Created, Running, Canceled, Completed
- name          VARCHAR,
- owner         VARCHAR(100),
- type          ENUM,   -- ModelChange, ConfigChange
- config        JSON,
- alloc_pct     DOUBLE,
- duration      INTEGER

Results:
- id            BIGINT (autogenerated),
- experiment_id BIGINT,
- metric        VARCHAR,
- value         NUMERIC
```

This represents a very basic data model for our setup. An actual A/B testing platform would possibly have more entities and columns. Given that there aren't that many experiments created frequently, and that the above data is quite tiny, we can use a simple RDBMS like Postgres for tracking this information.

## Deep dive

<figure class="blog-fig">
  <img src="/assets/images/ab-testing-deep-dive.png">
  <figcaption>Figure 1. Deep dive into system architecture</figcaption>
</figure>

Figure 1. above shows a detailed version of the system architecture. It shows 2 user journeys: (1) the way in which a developer creates and pushes a new experiment in the system, and (2) the way in which an end-user interacts with the service within an experiment. For protocol, it is sufficient to use HTTP REST endpoints for communication.

Upon requesting creation of a new experiment by a developer, a central config pusher service is called. This service first writes this request into an experiments DB. It also updates the load balancer config to let it know about this new experiment. The load balancer uses this information to assign users to this new experiment (based on `alloc_pct`) if they are currently not enrolled in any. The pusher then pushes the details about this new experiment to the server nodes hosting the service. At this point, the experiment setup is complete and the pusher service responds back to the caller with an `experiment_id` for the newly created experiment. The server nodes also register with the logs aggregation service informing it about the creation of this new experiment, the metrics to track for it, etc.

As an alternative to this design, one can also consider a more decoupled setup where the config pusher service interacts with the LB and server nodes via queues that sit in between. However, given that new experiments aren't created that often, this decoupling adds unnecessary complexity, and so we opt for the above approach.

On the end-user side, the device is already aware of the experiment it is in (or if it is part of the 'control' group). It attaches this information in all requests it makes to the service, shown via the `experiment_id=123` parameter in the diagram above. The load balancer balances the end-user requests over a set of stateless web servers. These web servers perform important (but auxiliary) tasks first, like authentication and request validation. Once the request is deemed good, it is parsed and a call to the backend service is made.

On the backend server, the request is first parsed. The passed in `experiment_id` is first mapped to a corresponding `thread ID`. The parsed request is then sent to that thread. There are a variety of ways implementing this. Just as an example, one is by writing to/reading from an in-memory queue. This thread holds the experiment model + config, and executes the experiment business logic. It then produces an output in a buffer, that is read from by a collector. The output is then returned to the caller.

Periodically, the end-user device also batches up user interactions (what the user tapped/clicked on, how long were they viewing, mouse maneuvering patterns, etc.) and sends this information to the service. The web servers recognize this different type of request, and forward this data to the logs aggregation service for enabling analytics. The logs aggregation uses this information and the metrics information from before, to compute quality metrics for the experiment and updates the experiments DB. A more decoupled approach is probably better here. We could add a queue in between the analytics service output and DB. This allows for scaling these 2 systems independently. In a real organization, these 2 services are very likely maintained by different teams, and so this queue serves as a nice contract between these systems. For brevity of diagram, the queue is not shown above.

## Further considerations

In a 45 minutes interview, it is impossible to cover all aspects mentioned above. Furthermore, your interviewer will possibly be interested in just a single aspect of the overall design, for example, just the data modeling part. Remember to always go wherever the interviewer wants you to go, but within that topic, drive the discussion.
If your design discussion is over, however, and you have a few minutes at the end to discuss extensions, consider the following topics:

1. Monitoring for the entire system. This includes system level monitoring, as well as at the individual thread level within each server.
2. Automatically stopping experiments, and reenrolling users in new experiments.
3. Dealing with server node failures.
4. In the above design, the models and configs for all experiments must together fit into memory. What if this is not the case?

Stopping an experiment automatically once it completes its duration is an interesting problem.
We need to do 2 cleanups: (1) remove the experiment from the load balancer config, and (2) stop the thread that corresponds to this experiment and reclaim its resources.
A simple way to implement this is to have a reaper service that periodically scans the experiments DB, and if it finds an experiment that has lapsed its duration, it performs the above cleanup. Care must be taken that this "periodic scan" is frequent enough so that the lapse in duration is bounded. For example, having the reaper scan run every 1 minute should be a good starting point.

Having the model + config collectively not fit in memory is a quite real possibility. One way to solve this issue is to instead have the models hosted on a _different_ set of servers, and have the service nodes make API calls to this other "models service" during inference. In this models service, each server will have in its memory just a single model, and a central bookkeeping service like ZooKeeper keeps track of what server hosts which model (and it also handles server failures, reinstantiations, network partitions, etc.). The service nodes also communicate with this ZooKeeper cluster to find out what machine the requested model is on.
