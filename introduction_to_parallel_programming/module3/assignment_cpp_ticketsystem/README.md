# Assignment: C++ ticket system

## New learnings
+ ```std::atomic_thread_fence(std::memory_order memOrder);``` is a synchronization primitive, that ensures that within the thread that is using this function, memory operations are done in a defined manner (defined by the ```std::memory_order memOrder```).
    - Even if thread instructions are executed in a supposedly sequential order, compilers and CPUs can execute them in parallel or in a different order, as long as the semantic of the thread function is not impaired.
        * Unfortunatly, compilers and CPU execution schedulers do not recognize the interdependencies between multiple threads, so in the example:
        ```
        #include <atomic>
        #include <thread>
        #include <iostream>

        std::atomic<bool> ready(false);
        int data = 0;

        void writer_thread() {
            data = 42;
            ready.store(true, std::memory_order::relaxed);
        }

        void reader_thread() {
            while (!ready.load(std::memory_order::relaxed));
            std::cout << data << std::endl;
        }

        int main() {
            std::thread t1(writer_thread);
            std::thread t2(reader_thread);

            t1.join();
            t2.join();

            return 0;
        }
        ```
        the reads and write of the ```data``` and the ```ready``` variables could be executed in different orders.  
        * The solution is to use the ```atomic_thread_fence```:
        ```
        #include <atomic>
        #include <thread>
        #include <iostream>

        std::atomic<bool> ready(false);
        int data = 0;

        void writer_thread() {
            data = 42;
            std::atomic_thread_fence(std::memory_order::release); // ensure that data is completly written before we set ready to true
            ready.store(true, std::memory_order::relaxed);
        }

        void reader_thread() {
            while (!ready.load(std::memory_order::relaxed));
            std::atomic_thread_fence(std::memory_order::acquire); // ensure that ready is readout before we print data
            std::cout << data << std::endl;
        }

        int main() {
            std::thread t1(writer_thread);
            std::thread t2(reader_thread);

            t1.join();
            t2.join();

            return 0;
        }
        ```
    - Main takeaways:
        * ```std::atomic_thread_fence``` only affects the calling thread
        * Is used to avoid race conditions 
+ Use ```std::this_thread::yield()``` instead of ```std::this_thread::sleep_for(std::chrono::seconds(1))``` for busy waiting. This will enable other threads to run during the wait and minimizes the CPU load while beeing as responsile as possible in the yielding thread.
+ ```std::future<int>``` ***do not*** have a copy constructor or copy assignment operator since this would invalidate them (they are linked to a specific ```std::promise<int>```). Therefore, you need to hand them over by reference via a ```std::ref(fut)```.