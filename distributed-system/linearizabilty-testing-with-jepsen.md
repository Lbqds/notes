最近看了如何使用Jepsen测试分布式系统，以线性一致性为例，简单做个总结。

Jepsen的github repo中包含了一个对etcd做测试的[教程](https://github.com/jepsen-io/jepsen/blob/main/doc/tutorial/index.md)，对于线性一致性的测试，无非就是对数据库做读写操作，同时引入一些错误，然后根据读写的结果来检查在错误发生的情况下系统的响应是否是正确的。

下面使用[Linearizability](https://cs.brown.edu/~mph/HerlihyW90/p463-herlihy.pdf)论文中第二节的系统模型以及线性一致性的定义。

整个测试的执行流程如下：

1. 首先定义SeqObj及其支持的调用
2. 使用多个Process并行的调用ConcObj，单个进程的请求必须得到响应之后才能进行下一次请求，同时引入一些错误
3. 根据请求响应的历史分析是否满足线性一致性

为了描述Jepsen是如何做的，下面分为几个部分，对于Jepsen的基本概念上述的教程中已经包括，这里不再赘述。

Jepsen版本: 0.2.1, Knossos版本: 0.3.7

## model

model就相当于上述论文中的SeqObj，model用于验证执行顺序操作之后是否满足一致性。对于Register model，Knossos的定义如下:

```clojure
(defrecord Register [value]
  Model
  (step [r op]
    (condp = (:f op)
      :write (Register. (:value op))
      :read  (if (or (nil? (:value op))     ; We don't know what the read was
                     (= value (:value op))) ; Read was a specific value
               r
               (inconsistent
                 (str (pr-str value) "≠" (pr-str (:value op)))))))

  Object
  (toString [r] (pr-str value)))
```

上面的定义可以看出如果读到的值不是上一次写入的值，就不满足一致性，这里的Register就是线性一致性模型。

除了Register之外，Knossos还提供了Mutex，CASRegister，FIFOQueue等model。

## generator

generator的目的是生成对ConcObj的调用请求，在当前版本的Jepsen中，为了便于generator的组合，将generator定义为immutable，定义如下:

```clojure
(defprotocol Generator
  (update [gen test context event]
          "Updates the generator to reflect an event having taken place.
          Returns a generator (presumably, `gen`, perhaps with some changes)
          resulting from the update.")

  (op [gen test context]
      "Obtains the next operation from this generator. Returns an pair
      of [op gen'], or [:pending gen], or nil if this generator is exhausted."))
```

`op`用于获取下一次调用的操作，有三种返回结果，`[op gen']`表示返回一个新的调用操作及新的generator; [:pending gen]表示当前generator挂起，比如在ConcurrentGenerator中，如果所有的线程都在等待调用结果，那么generator就会挂起; 返回`nil`表示generator已经生成所有的调用操作。

`update`用于更新当前的generator，由于generator是immutable的，所以update会返回一个新的generator。

`op`和`update`都有一个context参数，在每个Jepsen test开始时都会创建一个新的context:

```clojure
(defn context
  "Constructs a new context from a test."
  [test]
  (let [threads (->> (range (:concurrency test))
                     (cons :nemesis))
        threads (.forked (Set/from ^Iterable threads))]
    {:time          0
     :free-threads  threads
     :workers       (->> threads
                         (c/map (partial c/repeat 2))
                         (c/map vec)
                         (into {}))}))
```

可以看到，context用于测试初始化时根据指定的concurrency参数分配线程个数，同时记录哪些线程当前是空闲的，generator生成的调用操作会分配给空闲线程；context还包括一个time字段，表示相对时间(单位nanoseconds)， 会随着调用的执行和完成更新(下面的interpreter会提到)。

在Jepsen中，clojure的Map，Function，Seq等都是generator，定义如下:

```clojure
(extend-protocol Generator
  nil
  (update [gen test ctx event] nil)
  (op [this test ctx] nil)

  clojure.lang.APersistentMap
  (update [this test ctx event] this)
  (op [this test ctx]
    (let [op (fill-in-op this ctx)]
      [op (if (= :pending op) this nil)]))

  clojure.lang.AFunction
  (update [f test ctx event] f)

  (op [f test ctx]
    (when-let [x (if (= 2 (first (util/arities (class f))))
                   (f test ctx)
                   (f))]
      (op [x f] test ctx)))

  clojure.lang.Delay
  (update [d test ctx event] d)

  (op [d test ctx]
    (op @d test ctx))

  clojure.lang.Seqable
  (update [this test ctx event]
    (when (seq this)
      (cons (update (first this) test ctx event) (next this))))

  (op [this test ctx]
    (when (seq this)
      (let [gen (first this)]
        (if-let [[op gen'] (op gen test ctx)]
          [op (if-let [nxt (next this)]
                (cons gen' (next this))
                gen')]

          (recur (next this) test ctx))))))
```

以[etcd测试](https://github.com/jepsen-io/etcd)中对VersionedRegister的测试为例，定义了三个Function作为generator:

```clojure
(defn r   [_ _] {:type :invoke, :f :read, :value [nil nil]})
(defn w   [_ _] {:type :invoke, :f :write, :value [nil (rand-int 5)]})
(defn cas [_ _] {:type :invoke, :f :cas, :value [nil [(rand-int 5) (rand-int 5)]]})
```

Jepsen中还提供了很多combinator用于构造更复杂的generator，比如`stagger`用于指定调用频率，`time-limit`用于指定时间限制，`mix`表示将多个generator混合在一起，每次从中随机选取一个。

更多的combinator定义在[generator.clj](https://github.com/jepsen-io/jepsen/blob/main/jepsen/src/jepsen/generator.clj)中。除此之外，在[independent.clj](https://github.com/jepsen-io/jepsen/blob/main/jepsen/src/jepsen/generator/interpreter.clj)中还定义了ConcurrentGenerator，ConcurrentGenerator会将指定的key附加到调用操作的value中，以Register为例，如果使用ConcurrentGenerator相当于同时对多个Register测试，会更容易暴露出系统的bug。

前面提到了，generator只是生成调用操作，具体操作调用的执行并将所有的操作保存为一个历史记录由interpreter来执行:

```clojure
(defn run!
  [test]
  (gen/init!)
  ; 创建context，通过spawn-worker启动worker(其中包括nemesis)，每一个worker有一个大小为1的ArrayBlockingQueue，这里的
  ; loop会将调用操作放入worker的队列中，worker会从队列里取出调用操作并执行，并将执行结果放入completions队列中，每个worker
  ; 在接收到响应之前不会执行下一次调用。
  (let [ctx         (gen/context test)
        worker-ids  (gen/all-threads ctx)
        completions (ArrayBlockingQueue. (count worker-ids))
        workers     (mapv (partial spawn-worker test completions
                                   (client-nemesis-worker))
                                   worker-ids)
        invocations (into {} (map (juxt :id :in) workers))
        gen         (->> (:generator test)
                         gen/friendly-exceptions
                         gen/validate)]
    (try+
      (loop [ctx            ctx
             gen            gen
             outstanding    0
             poll-timeout   0
             history        (transient [])]
        ; 是否已经有完成的操作，如果有的话更新context time并将更新调用操作的响应时间(可以用于性能分析)，
        ; 同时更新空闲的线程，并更新generator，然后将这个调用结果放入history中
        (if-let [op' (.poll completions poll-timeout TimeUnit/MICROSECONDS)]
          (let [thread (gen/process->thread ctx (:process op'))
                time    (util/relative-time-nanos)
                op'     (assoc op' :time time)
                ctx     (assoc ctx
                              :time         time
                              :free-threads (.add ^Set (:free-threads ctx)
                                                  thread))
                gen     (gen/update gen test ctx op')
                ctx     (if (or (= :nemesis thread) (not= :info (:type op')))
                          ctx
                          (update ctx :workers assoc thread
                                  (gen/next-process ctx thread)))
                history (if (goes-in-history? op')
                          (conj! history op')
                          history)]
            (recur ctx gen (dec outstanding) 0 history))

          ; 没有已完成的调用操作，更新context time，并调用generator尝试获取下一个调用操作
          (let [time        (util/relative-time-nanos)
                ctx         (assoc ctx :time time)
                [op gen']   (gen/op gen test ctx)]
            (condp = op
             ; 如果generator op返回nil，表示所有的调用操作都已经执行，现在需要等待未收到响应的调用操作(outstanding)
              nil (if (pos? outstanding)
                    (recur ctx gen outstanding (long max-pending-interval)
                           history)
                    (do (doseq [[thread queue] invocations]
                          (.put ^ArrayBlockingQueue queue {:type :exit}))
                        ; Wait for exit
                        (dorun (map (comp deref :future) workers))
                        (persistent! history)))

              :pending (recur ctx gen outstanding (long max-pending-interval)
                              history)

              ; 根据调用操作的time来判断当前是否可以执行操作
              (if (< time (:time op))
                (do
                    (recur ctx gen outstanding
                           (long (/ (- (:time op) time) 1000))
                           history))

                (let [thread (gen/process->thread ctx (:process op))
                      ; 为调用操作分配一个线程，并更新的context的空闲线程
                      _ (.put ^ArrayBlockingQueue (get invocations thread) op)
                      ctx (assoc ctx
                                 :time (:time op) ; Use time instead?
                                 :free-threads (.remove
                                                 ^Set (:free-threads ctx)
                                                 thread))
                      ; 更新generator并将调用操作放入history中
                      gen' (gen/update gen' test ctx op)
                      history (if (goes-in-history? op)
                                (conj! history op)
                                history)]
                  (recur ctx gen' (inc outstanding) 0 history)))))))

      (catch Throwable t
        (info "Shutting down workers after abnormal exit")
        (dorun (map (comp future-cancel :future) workers))
        (loop [unfinished workers]
          (when (seq unfinished)
            (let [{:keys [in future] :as worker} (first unfinished)]
              (if (future-done? future)
                (recur (next unfinished))
                (do (.offer ^java.util.Queue in {:type :exit})
                    (recur unfinished))))))
        (throw t)))))
```

这里需要注意的一点是，如果执行调用的Process(注意不是thread，Process由thread执行)crash，那么Jepsen不知道这个操作是否成功，也就是说这个操作还处于中间状态，在生成的timeline中会占用整个时间线，具体可以参考[这里](https://github.com/jepsen-io/jepsen/blob/main/doc/tutorial/06-refining.md)和[这里](https://medium.com/appian-engineering/chaos-testing-a-distributed-system-with-jepsen-part-iii-923721504130)

## nemesis

为了测试分布式系统的高可用性以及容错性，可以通过nemesis引入一些错误，比如对于etcd线性一致性测试来说，我们可以引入网络分区，网络延时等错误，并测试当网络分区出现时，在不同的分区是否能请求成功。

Jepsen nemesis提供了多种引入错误的方法，比如网络分区可以通过partitioner来完成:

```clojure
(defn partitioner
  ([] (partitioner nil))
  ([grudge]
   (reify Nemesis
     (setup! [this test]
       (net/heal! (:net test) test)
       this)

     (invoke! [this test op]
       (case (:f op)
         :start (let [grudge (or (:value op)
                                 (if grudge
                                   (grudge (:nodes test))
                                   (throw (IllegalArgumentException.
                                            (str "Expected op " (pr-str op)
                                                 " to have a grudge for a :value, but none given.")))))]
                  (net/drop-all! test grudge)
                  (assoc op :value [:isolated grudge]))
         :stop  (do (net/heal! (:net test) test)
                    (assoc op :value :network-healed))))

     (teardown! [this test]
       (net/heal! (:net test) test)))))
```

```clojure
(defn partition-random-halves
  []
  (partitioner (comp complete-grudge bisect shuffle)))
```

比如对于五个节点的集群，节点编号为[1, 2, 3, 4, 5]，`partition-random-halves`会随机选择网络分区的节点，一种可能的情况为:

```
{2: [1, 4, 5]}  => 断开节点2跟节点1,4,5之间的连接，下同
{3: [1, 4, 5]}
{1: [2, 3]}
{4: [2, 3]}
{5: [2, 3]}
```

`Net`提供了断开网络，网络延时，丢包的接口，默认的实现中，网络断开由iptables实现，即将相关的数据包过滤掉，网络恢复时再将过滤规则删除，网络延时和丢包由`tc`完成。

## checker

当interpreter执行完成之后，会把整个执行历史保存下来，供checker检查是否满足一致性要求。以Register为例，interpreter返回的是多个线程并行调用读写的历史记录。根据上述论文中的模型，一个操作由对应的调用请求(invoke)和调用响应(resp)组成，整个历史中的操作存在一个偏序关系<<sub>h</sub>，即两个操作op1和op2，`invoke(op2)`发生在`resp(op1)`之后，则op1 <<sub>h</sub> op2成立。不满足<<sub>h</sub>关系的操作成为是并行的。

如果想要验证整个执行历史是否满足model中定义的一致性，那么只需要对整个历史记录中的所有并行的操作进行排列组合，同时满足历史记录中原有的<<sub>h</sub>关系，将所有可能的操作序列逐次执行，如果存在某个操作序列满足model的定义，那么model中的一致性就可以满足。Jepsen checker中的[wgl](http://www.cs.cmu.edu/~wing/publications/WingGong93.pdf)算法就是就是这么做的，当然如果整个历史记录非常大，对所有的并行操作做排列组合不太现实，所以Jepsen的实现中采用了回溯算法，即按照历史记录依次执行，如果执行结果不满足model的定义，则回溯到上一个linearizable的点，然后尝试新的排列组合(下面会详细描述)。

为了便于下面的描述，我们先看两个例子，对于Register测试，有下面两个操作历史(后面的注释为op id):

history1:

```clojure
(def history1
  [{:type :invoke, :f :write, :value 0, :process 0}   ; 0
   {:type :invoke, :f :write, :value 1, :process 1}   ; 1
   {:type :invoke, :f :write, :value 2, :process 2}   ; 2
   {:type :invoke, :f :read, :value nil, :process 3}  ; 3
   {:type :ok, :f :read, :value 1, :process 3}        ; 3
   {:type :ok, :f :write, :value 1, :process 1}       ; 1
   {:type :ok, :f :write, :value 2, :process 2}       ; 2
   {:type :ok, :f :write, :value 0, :process 0}       ; 0
   ])
```

history2:

```clojure
(def history2
  [{:type :invoke, :f :write, :value 0, :process 0}   ; 0
   {:type :ok, :f :write, :value 0, :process 0}       ; 0
   {:type :invoke, :f :write, :value 1, :process 1}   ; 1
   {:type :invoke, :f :write, :value 2, :process 2}   ; 2
   {:type :ok, :f :write, :value 1, :process 1}       ; 1
   {:type :invoke, :f :read, :value nil, :process 3}  ; 3
   {:type :ok, :f :read, :value 0, :process 3}        ; 3
   {:type :ok, :f :write, :value 2, :process 2}       ; 2
   ])
```

显而易见，history1满足线性一致性，而history2不满足线性一致性(op3的读操作不可能读到0)。

下面详细介绍Jepsen中wgl算法的实现(在Knossos库中)，感兴趣的也可以去看[wgl算法论文](http://www.cs.cmu.edu/~wing/publications/WingGong93.pdf):

```clojure
(defn check
  [model history state]
  (let [{:keys [history kindex-eindex]} (history/preprocess history)
        _ (swap! state assoc :indices kindex-eindex)
        history (->> history
                     with-entry-ids
                     (mapv map->Op))
        ; 首先尝试缓存所有可能的执行结果，memo会将所有非重复的调用排列组合重复执行，
        ; 直至计算出所有可能的结果，相当于建立一个状态转换图，后面再去执行响应的请求时
        ; 只需要直接查找即可，相关的代码在knossos/src/knossos/model/memo.clj中，
        ; 前面也提到了，对于比较长的历史记录，我们不可能计算出所有可能的结果，因此如果
        ; 在计算的过程中如果发现可能的结果查过1024个，就不会再继续做缓存
        {:keys [model history]} (memo model history)
        n           (max (max-entry-id history) 0)
        ; dll-history将历史记录构造为一个双向链表，对于每个invoke操作，还有一个额外的
        ; 指针指向resp操作记录，这个双向链表提供了lift!和unlift!两个方法，lift!将对应
        ; 的操作(包括调用和响应)从链表中删除，unlift!将对应的操作插入原来的位置。lift!和
        ; unlift!主要用于重试新的排列组合
        head-entry  (dllh/dll-history history)
        linearized  (BitSet. n)
        cache       (doto (HashSet.)
                      (.add (cache-config (BitSet.) model nil)))
        ; invoke操作栈，当不满足一致性要求时，弹出栈顶元素，然后unlift!到链表中，相当于
        ; 回溯到上一个线性化记录，注意这里会将操作和执行的结果同时缓存起来
        calls       (ArrayDeque.)]
    (loop [s              model
           ^Node entry    (.next head-entry)]
        (cond
          (not (:running? @state))
          {:valid? :unknown
           :cause  (:cause @state)}

          (not (.next head-entry))
          {:valid? true
           :model  (memo/unwrap s)}

          true
          (let [op ^Op (.op entry)
                type (.type op)]
            (condp identical? type
              :invoke
              ; 这里的执行如果前面的memo阶段已经做了缓存的话，直接查找就可以了，
              ; memo的目的就是为了避免多次执行，减少checker的时间
              (let [s' (model/step s op)]
                (if (model/inconsistent? s')
                  ; 如果当前的invoke操作不满足model，则尝试下一个操作记录
                  (recur s (.next entry))

                  ; 如果满足model，则将这个操作记为已线性化，然后从链表中删除
                  ; 这个invoke操作和其对应的resp记录
                  (let [entry-id    (.entry-id op)
                        linearized' ^BitSet (.clone linearized)
                        _           (.set linearized' entry-id)
                        new-config? (.add cache
                                          (cache-config linearized' s' op))]
                    (if (not new-config?)
                      ; 如果之前已经对这个序列做了线性化操作，则直接跳过
                      (let [entry (.next entry)]
                        (recur s entry))

                      (let [_           (.addFirst calls [entry s])
                            s           s'
                            _           (.set linearized entry-id)
                            _           (dllh/lift! entry)
                            ; 这里注意，重新从head-entry执行，即尝试执行之前可能存在的
                            ; 不满足model的invoke操作
                            entry       (.next head-entry)]
                        (recur s entry))))))

              ; resp操作表明我们还没有对其invoke操作做线性化，因为对已经线性化的操作我们已经
              ; 将其resp删除，如果当前的calls调用栈是空的话，说明整个历史记录不满足model；
              ; 如果calls调用栈不为空，则弹出栈顶的invoke，尝试新的排列组合。
              :ok
              (if (.isEmpty calls)
                (invalid-analysis history cache state)

                ; 弹出calls的栈顶元素，回溯到上一个序列化的点，尝试新的排列组合
                (let [[^INode entry s]  (.removeFirst calls)
                      op                ^Op (.op entry)
                      _                 (.set linearized ^long (.entry-id op)
                                              false)
                      _                 (dllh/unlift! entry)
                      entry             (.next entry)]
                  (recur s entry)))

              :info
              {:valid? true
               :model  (memo/unwrap s)}))))))
```

如果上述算法不好理解的话，可以拿上面的history1为例，一步一步执行看看是怎么做回溯的。对于这个算法的正确性描述，感兴趣的可以看原始的论文。如果发现当前的历史不满足model的定义时，会通过`invalid-analysis`生成一个分析结果。感兴趣的同学也可以执行如下代码来查看上面定义的两个历史记录history1和history2是否满足线性一致性要求:

```clojure
(defn -main
  [& args]
  (let [model (kmodel/cas-register nil)
        res1 (knossos.wgl/analysis model history1)
        res2 (knossos.wgl/analysis model history2)]
    (pprint res1)
    (pprint res2)))
```

执行结果如下，执行结果表明history1是满足线性一致性的，history2是不满足的:

```clojure
{:valid? true, :model {:value 2}, :analyzer :wgl}
{:valid? false,
 :op {:process 3, :type :ok, :f :read, :value 0, :index 6},
 :previous-ok {:process 1, :type :ok, :f :write, :value 1, :index 4},
 :final-paths
 #{[{:op {:process 2, :type :ok, :f :write, :value 2, :index 7},
     :model {:value 2}}
    {:op {:process 3, :type :ok, :f :read, :value 0, :index 6},
     :model {:msg "can't read 0 from register 2"}}]
   [{:op {:process 1, :type :ok, :f :write, :value 1, :index 4},
     :model {:value 1}}
    {:op {:process 3, :type :ok, :f :read, :value 0, :index 6},
     :model {:msg "can't read 0 from register 1"}}]
   [{:op {:process 1, :type :ok, :f :write, :value 1, :index 4},
     :model {:value 1}}
    {:op {:process 2, :type :ok, :f :write, :value 2, :index 7},
     :model {:value 2}}
    {:op {:process 3, :type :ok, :f :read, :value 0, :index 6},
     :model {:msg "can't read 0 from register 2"}}]},
 :analyzer :wgl}
```



除了wgl算法之外，Knossos还实现了[LinearizabiltyTesting](http://www.cs.ox.ac.uk/people/gavin.lowe/LinearizabiltyTesting/paper.pdf)论文中描述的算法，对于wgl算法来说计算量会更少(我还没看，后面抽时间做一个跟wgl算法做一个对比)。

### 其他

除了验证之外，Jepsen还提供了timeline，perf等功能，timeline生成一个html页面，将所有的操作历史的时间线标记出来。perf会根据调用和响应时间去分析延时和吞吐量，并通过gnuplot画出来。

一些参考资料:

1. wgl论文: http://www.cs.cmu.edu/~wing/publications/WingGong93.pdf
2. LinearizabilityTesting论文: http://www.cs.ox.ac.uk/people/gavin.lowe/LinearizabiltyTesting/paper.pdf
3. [Linearizability: A Correctness Condition for Concurrent Objects](https://cs.brown.edu/~mph/HerlihyW90/p463-herlihy.pdf)
4. [Chaos Testing a Distributed System with Jepsen](https://medium.com/appian-engineering/chaos-testing-a-distributed-system-with-jepsen-2ae4a8bdf4e5)

