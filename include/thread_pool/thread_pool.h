#pragma once

#include <atomic>
#include <barrier>
#include <concepts>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <semaphore>
#include <thread>
#include <type_traits>
#ifdef __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#endif

#include "thread_pool/thread_safe_queue.h"

namespace dp {
    namespace details {

#ifdef __cpp_lib_move_only_function
        using default_function_type = std::move_only_function<void()>;
#else
        using default_function_type = std::function<void()>;
#endif
    }  // namespace details

    template <typename FunctionType = details::default_function_type,
              typename ThreadType = std::jthread>
        requires std::invocable<FunctionType> &&
                 std::is_same_v<void, std::invoke_result_t<FunctionType>>
    class thread_pool {
      public:
        explicit thread_pool(
            const unsigned int &number_of_threads = std::thread::hardware_concurrency())
            : tasks_(number_of_threads) {
            std::size_t current_id = 0;
            for (std::size_t i = 0; i < number_of_threads; ++i) {
                priority_queue_.push_back(size_t(current_id));
                try {
                    threads_.emplace_back([&, id = current_id](const std::stop_token &stop_tok) {
                        do {
                            // wait until signaled
                            tasks_[id].signal.acquire();

                            do {
                                // 从tasks_[id]的任务队列的front取任务执行
                                // invoke the task
                                while (auto task = tasks_[id].tasks.pop_front()) {
                                    try {
                                        pending_tasks_.fetch_sub(1, std::memory_order_release);
                                        std::invoke(std::move(task.value()));
                                    } catch (...) {
                                    }
                                }
                                // 从tasks_[id]的下一个task到前一个task的任务队列尾，偷任务执行
                                // 比如id=5, tasks_.size()=8，那就从6，7，0，1，2，3，4从偷任务
                                // 这里有点小问题(也不能说是一个问题，只是这个实现很奇怪), 如果steal 6 然后break，下一轮当前线程的队列还是没有任务，那么还会steal 6
                                // try to steal a task
                                for (std::size_t j = 1; j < tasks_.size(); ++j) {
                                    const std::size_t index = (id + j) % tasks_.size();
                                    if (auto task = tasks_[index].tasks.steal()) {
                                        // steal a task
                                        pending_tasks_.fetch_sub(1, std::memory_order_release);
                                        std::invoke(std::move(task.value()));
                                        // stop stealing once we have invoked a stolen task
                                        break;
                                    }
                                }

                            } while (pending_tasks_.load(std::memory_order_acquire) > 0);

                            priority_queue_.rotate_to_front(id);

                        } while (!stop_tok.stop_requested());
                    });
                    // increment the thread id
                    ++current_id;

                } catch (...) { // 可能是创建thread分配内存异常
                    // catch all

                    // remove one item from the tasks
                    tasks_.pop_back();

                    // remove our thread from the priority queue
                    std::ignore = priority_queue_.pop_back();
                }
            }
        }

        ~thread_pool() {
            // stop all threads
            for (std::size_t i = 0; i < threads_.size(); ++i) {
                threads_[i].request_stop();
                tasks_[i].signal.release();
                threads_[i].join();
            }
        }

        /// thread pool is non-copyable
        thread_pool(const thread_pool &) = delete;
        thread_pool &operator=(const thread_pool &) = delete;

        /**
         * @brief Enqueue a task into the thread pool that returns a result.
         * @details Note that task execution begins once the task is enqueued.
         * @tparam Function An invokable type.
         * @tparam Args Argument parameter pack
         * @tparam ReturnType The return type of the Function
         * @param f The callable function
         * @param args The parameters that will be passed (copied) to the function.
         * @return A std::future<ReturnType> that can be used to retrieve the returned value.
         */
        template <typename Function, typename... Args,
                  typename ReturnType = std::invoke_result_t<Function &&, Args &&...>>
            requires std::invocable<Function, Args...>
        [[nodiscard]] std::future<ReturnType> enqueue(Function f, Args... args) {
#ifdef __cpp_lib_move_only_function
            // we can do this in C++23 because we now have support for move only functions
            std::promise<ReturnType> promise;
            auto future = promise.get_future();
            auto task = [func = std::move(f), ... largs = std::move(args),
                         promise = std::move(promise)]() mutable {
                try {
                    if constexpr (std::is_same_v<ReturnType, void>) {
                        func(largs...);
                        promise.set_value();
                    } else {
                        promise.set_value(func(largs...));
                    }
                } catch (...) {
                    promise.set_exception(std::current_exception());
                }
            };
            enqueue_task(std::move(task));
            return future;
#else
            /*
             * use shared promise here so that we don't break the promise later (until C++23)
             *
             * with C++23 we can do the following:
             *
             * std::promise<ReturnType> promise;
             * auto future = promise.get_future();
             * auto task = [func = std::move(f), ...largs = std::move(args),
                              promise = std::move(promise)]() mutable {...};
             */
            auto shared_promise = std::make_shared<std::promise<ReturnType>>();
            auto task = [func = std::move(f), ... largs = std::move(args),
                         promise = shared_promise]() {
                try {
                    if constexpr (std::is_same_v<ReturnType, void>) {
                        func(largs...);
                        promise->set_value();
                    } else {
                        promise->set_value(func(largs...));
                    }

                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            };

            // get the future before enqueuing the task
            auto future = shared_promise->get_future();
            // enqueue the task
            enqueue_task(std::move(task));
            return future;
#endif
        }

        /**
         * @brief Enqueue a task to be executed in the thread pool that returns void.
         * @tparam Function An invokable type.
         * @tparam Args Argument parameter pack for Function
         * @param func The callable to be executed
         * @param args Arguments that will be passed to the function.
         */
        template <typename Function, typename... Args>
            requires std::invocable<Function, Args...> &&
                     std::is_same_v<void, std::invoke_result_t<Function &&, Args &&...>>
        void enqueue_detach(Function &&func, Args &&...args) {
            enqueue_task(
                std::move([f = std::forward<Function>(func),
                           ... largs = std::forward<Args>(args)]() mutable -> decltype(auto) {
                    // suppress exceptions
                    try {
                        std::invoke(f, largs...);
                    } catch (...) {
                    }
                }));
        }

        [[nodiscard]] auto size() const { return threads_.size(); }

      private:
        template <typename Function>
        void enqueue_task(Function &&f) {
            auto i_opt = priority_queue_.copy_front_and_rotate_to_back();
            if (!i_opt.has_value()) {
                // would only be a problem if there are zero threads
                return;
            }
            auto i = *(i_opt);
            pending_tasks_.fetch_add(1, std::memory_order_relaxed);
            tasks_[i].tasks.push_back(std::forward<Function>(f));
            // 这里视乎也有小问题,每push一个task就release一次,但是线程只会在当前队列没用任务且steal的时候才acquire一次
            // 虽然是binary_semaphore，但是测试发现如果release多次，就可以acquire多次
            // 这样导致，如果被release了多次后，如果队列没有任务且steal失败后，还是可以acquire成功
            // https://stackoverflow.com/questions/75556262/c20-binary-semaphore-goes-over-max
            tasks_[i].signal.release();
        }

        struct task_item {
            dp::thread_safe_queue<FunctionType> tasks{};
            // 当signal为0时线程等待
            // 相当于完成 std::unique_lock lk(mtx); cv.wait(lk, [] { return !thread_safe_queue.empty(); });
            std::binary_semaphore signal{0};
        };

        std::vector<ThreadType> threads_;  // 线程池里的线程
        std::deque<task_item> tasks_;  // 线程池里的线程对应的任务队列, 但这里并没用用到dequeue的特性，用std::vector应该也可以
        // push进来的任务给哪个线程的队列: thread_id = priority_queue_.front()
        // 什么时候priority_queue会发生变化: 当一个线程执行完自己队列中的任务，并且steal完其他队列的任务，priority_queue_.rotate_to_front(thread_id);
        // 相当于将任务优先放到那些刚执行完一轮任务(while(my_task) + while(steal_task)的线程的任务队列中
        dp::thread_safe_queue<std::size_t> priority_queue_;
        std::atomic_int_fast64_t pending_tasks_{};  // 记录总的任务数, 线程取出任务后减1,线程池增加任务后加1
    };

    /**
     * @example mandelbrot/source/main.cpp
     * Example showing how to use thread pool with tasks that return a value. Outputs a PPM image of
     * a mandelbrot.
     */
}  // namespace dp
