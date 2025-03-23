#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <algorithm>
#include <optional>

// Datamodell of the user
struct User {
    int id;
    std::string name;
};

// Repository-class to secure the data integrity
class UserRepository {
public:
    void AddUser(const User& User) {
        std::lock_guard<std::shared_mutex> lock(dataMtx); // exclusive lock due to writing the data
        allUsers.push_back(User);
        std::cout << "Added User '" << User.name << std::endl;
    }

    // getting user by id
    std::optional<User> getUser(int id) {
        std::shared_lock<std::shared_mutex> lock(dataMtx);
        auto it = std::find_if(allUsers.begin(), allUsers.end(),
            [id](const User& User) { return User.id == id; });

        if (it != allUsers.end()) {
            return *it;
        } else {
            return std::nullopt;
        }
    }

    // get all users as vector
    std::vector<User> getAll() {
        std::shared_lock<std::shared_mutex> lock(dataMtx); // shared lock - multiple threads can access the data at the same time but no one of them can write to the data
        return allUsers;
    }

private:
    std::vector<User> allUsers;
    std::shared_mutex dataMtx; // secure writing but enable multiple reads at the same time
};

/* Thread functions */
void addingUser(UserRepository& repo, User User) {
    repo.AddUser(User);
}

void getAllUsers(UserRepository& repo, int id) {
    auto User = repo.getUser(id);
    if (User) {
        std::cout << "User gefunden: " << User->name << std::endl;
    } else {
        std::cout << "User mit ID " << id << " nicht gefunden." << std::endl;
    }
}

int main() {
    // create data management
    UserRepository repo;

    // user producing threads
    std::thread t1(addingUser, std::ref(repo), User{1, "Alice"});
    std::thread t2(addingUser, std::ref(repo), User{2, "Bob"});

    // readout threads
    std::thread t3(getAllUsers, std::ref(repo), 1);
    std::thread t4(getAllUsers, std::ref(repo), 3);
    
    // join all threads
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    return EXIT_SUCCESS;
}