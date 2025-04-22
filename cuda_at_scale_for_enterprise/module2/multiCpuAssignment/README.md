# Assignment Remarks
+ The main issue with the assignment is that the synchronization with the lock files does not work properly out of the box. You need to interpret the files as a not-acquired lock and the absence of the files can be interpreted as the currently acquirement of the lock.
    - The python application is okay, the only need to make the synchronization work was the `.cu` file
+ The branching as well as the semi-algebraic solution of the cuda kernel code are correct in my solution