#include <iostream>
#include <thread>
#include <deque>
#include <mutex>
#include <chrono>
#include <condition_variable>

using std::deque;
std::mutex mu; // buffer secureing
std::mutex cout_mu; // printout securing
std::condition_variable cond; // notification mechanism

/**
* @brief this is the actual implementation of the producer-consumer pattern, since it describes the message queue
*/
class Buffer
{
public:
    void add(int num) {
        while (true) {
            std::unique_lock<std::mutex> locker(mu);
            // the predicate is evaluated at the first call of wait() and whenever a .notify_all() or .notify_one() call is executed on the condition_variable
            cond.wait( // wait blocks the mutex, checks the predicate and if the predicate evaluates to false, the mutex gets UNLOCKED (other threads can lock it) and this thread blocks until the condition_variable is notified.
                locker, // mutex to block and check
                [this](){
                    // CAUTION: You need to use the data that the mutex should secure within the predicate - during the predicate check, this thread will have exclusive access to the data, secured by the mutex
                    return buffer_.size() < size_; // check predicate and if it evaluates to true go ahead with the code, if it evaluates to false, the code will sleep until the condition_variable is notified
                }
            ); // Whats nice about condition_variables is, is that the waiting thread acts as if the mutex is permanently locked
            buffer_.push_back(num);
            locker.unlock();
            cond.notify_all();
            return;
        }
    }
    int remove() {
        while (true)
        {
            std::unique_lock<std::mutex> locker(mu);
            cond.wait(locker, [this](){return buffer_.size() > 0;});
            int back = buffer_.back();
            buffer_.pop_back();
            locker.unlock();
            cond.notify_all();
            return back;
        }
    }
    Buffer() {}
private:
    deque<int> buffer_; // due to the predicate function of void add(int num) the deque can maximally extend to 10 elements
    const unsigned int size_ = 10;
};

class Producer
{
public:
    Producer(Buffer* buffer, std::string name)
    {
        this->buffer_ = buffer;
        this->name_ = name;
    }
    void run() {
        while (true) {
            int num = std::rand() % 100;
            buffer_->add(num); // access the secured buffer resource
            cout_mu.lock();
            int sleep_time = rand() % 100;
            std::cout << "Name: " << name_ << "   Produced: " << num << "   Sleep time: " << sleep_time << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            cout_mu.unlock();
        }
    }
private:
    Buffer *buffer_;
    std::string name_;
};

class Consumer
{
public:
    Consumer(Buffer* buffer, std::string name)
    {
        this->buffer_ = buffer;
        this->name_ = name;
    }
    void run() {
        while (true) {
            int num = buffer_->remove(); // access the secured buffer resource
            cout_mu.lock();
            int sleep_time = rand() % 100;
            std::cout << "Name: " << name_ << "   Consumed: " << num << "   Sleep time: " << sleep_time << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            cout_mu.unlock();
        }
    }
private:
    Buffer *buffer_;
    std::string name_;
};

int main() {
    Buffer b; // dependency injection - good, losely coupled C++ designed
    Producer p1(&b, "producer1");
    Producer p2(&b, "producer2");
    Producer p3(&b, "producer3");
    Consumer c1(&b, "consumer1");
    Consumer c2(&b, "consumer2");
    Consumer c3(&b, "consumer3");

    std::thread producer_thread1(&Producer::run, &p1);
    std::thread producer_thread2(&Producer::run, &p2);
    std::thread producer_thread3(&Producer::run, &p3);

    std::thread consumer_thread1(&Consumer::run, &c1);
    std::thread consumer_thread2(&Consumer::run, &c2);
    std::thread consumer_thread3(&Consumer::run, &c3);

    producer_thread1.join();
    producer_thread2.join();
    producer_thread3.join();
    consumer_thread1.join();
    consumer_thread2.join();
    consumer_thread3.join();

    getchar();
    return 0;
}