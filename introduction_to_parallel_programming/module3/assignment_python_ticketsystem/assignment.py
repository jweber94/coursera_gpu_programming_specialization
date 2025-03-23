# Based on RealPython Threading Example page at https://realpython.com/intro-to-python-threading/ and
#   Python.org _thread library documentation at
#   https://docs.python.org/3/library/_thread.html?highlight=_thread#module-_thread
import logging
import random
import sys
import time
from threading import Thread, Lock, Event
from core import Core
from datetime import datetime

# This is the repository pattern
class TicketManager():
    def __init__(self):
        self.event = Event()
        self.lck = Lock()
        self.currentTicketNum = 0

    def waitForTicket(self, timeout=1) -> int:
        self.event.wait(timeout)
        return self.currentTicketNum

    '''
        The only way to increment the ticket is to set a ticket as done - there is no intelligence based on a timeout or something since this would be much more complext then a basic assignment in this course
    '''
    def setTicketDone(self, ticketNum) -> bool:
        #if self.currentTicketNum == 0: # early stop if we are in an invalid operating state
        #    #print("ERROR: You can not set a ticket done if the ticket manager did not start its operation.")
        #    return False

        self.lck.acquire() # secure current ticket acess
        ret = False
        if self.currentTicketNum == ticketNum: # check if the thread has the current ticket number
            self.currentTicketNum += 1
            ret = True
            self.event.clear() # reset the event to have all other threads waiting again
        else:
            ret = False
        self.lck.release()

        self.event.set() # ask for the next thread to get its ticket done
        return ret

    def getCurrentTicketNum(self) -> int:
        return self.currentTicketNum

    def startTicketSystem(self):
        #self.currentTicketNum += 1
        self.event.set()


def execute_ticketing_system_participation(ticket_number, part_id, shared_variable):
    output_file_name = "output-" + part_id + ".txt"
    # NOTE: Do not remove this print statement as it is used to grade assignment,
    # so it should be called by each thread
    print("Thread retrieved ticket number: {} started".format(ticket_number), file=open(output_file_name, "a"))
    time.sleep(random.randint(0, 10))
    # wait until your ticket number has been called
    terminate = False
    while not terminate:
        shared_variable.waitForTicket(3) # wait for ticket with timeout to ensure that if you wait to long, you try to get your ticket done (this is the edge case that you are the last ticket and the timing of the event trigger was before you got to sleep here)
        if shared_variable.setTicketDone(ticket_number):
            terminate = True
        #else:
        #    print("Not my ticket - try again")


    # output your ticket number and the current time
    print("Ticket: " + str(ticket_number) + "; Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # NOTE: Do not remove this print statement as it is used to grade assignment,
    # so it should be called by each thread

    print("Thread with ticket number: {} completed".format(ticket_number), file=open(output_file_name, "a"))
    return 0


class Assignment(Core):

    USERNAME = "Jens"
    active_threads = []
    ticketManager = TicketManager()

    def __init__(self, args):
        self.num_threads = 1
        self.args_conf_list = [['-n', 'num_threads', 1, 'number of concurrent threads to execute'],
                                ['-u', 'user', None, 'the user who is turning in the assignment, needs  to match the '
                                                    '.user file contents'],
                               ['-p', 'part_id', 'test', 'the id for the assignment, test by default']]
        super().__init__(self.args_conf_list)
        super().parse_args(args=args)
        _format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=_format, level=logging.INFO,
                            datefmt="%H:%M:%S")

    def run(self):
        output_file_name = "output-" + self.part_id + ".txt"
        open(output_file_name, 'w').close()
        if self.test_username_equality(self.USERNAME):
            sleeping_time = 0
            # create all threads
            for index in range(self.num_threads):
                logging.info("Assignment run    : create and start thread %d.", index)
                # This is where you will start a thread that will participate in a ticketing system
                # have the thread run the execute_ticketing_system_participation function
                threadToStart = Thread(group=None, target=execute_ticketing_system_participation, args=(index, self.part_id, self.ticketManager)) # index + 1 is the ticket number - there is no 0th ticket
                threadToStart.start()
                self.active_threads.append(threadToStart)

                # Threads will be given a ticket number and will wait until a shared variable is set to that number
                # The code will also need to know when all threads have completed their work
                sleeping_time += 1
            time.sleep(sleeping_time)
            #time.sleep(1)
            self.manage_ticketing_system() # start the ticket system to operate

            # waiting for all threads to complete - the order does not matter - this will block until the last thread has finished since the array is fixed when this for loop is executed
            for thread in self.active_threads:
                thread.join() # an alternative could be the .is_alive() method of the thread object

            logging.info("Assignment completed all running threads.")
            return 0
        else:
            logging.error("Assignment had an error your usernames not matching. Please check code and .user file.")
            return 1

    def manage_ticketing_system(self):
        # increment a ticket number shared by a number of threads and check that no active threads are running
        self.ticketManager.startTicketSystem()
        return 0


if __name__ == "__main__":
    assignment = Assignment(args=sys.argv[1:])
    exit_code = assignment.run()
    sys.exit(exit_code)
