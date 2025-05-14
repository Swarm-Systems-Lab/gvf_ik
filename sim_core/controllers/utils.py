"""
"""

__all__ = ["CircularBuffer"]

#######################################################################################

class CircularBuffer:
    def __init__(self, capacity, max_period):
        self.capacity = capacity
        self.max_period = max_period
        self.max_period_flag = False

        self.buffer = [None] * capacity
        self.timestamps = [None] * capacity
        self.head = 0  # Points to the oldest element
        self.tail = 0  # Points to the next insertion point
        self.size = 0

    def is_full(self):
        return self.size == self.capacity

    def is_empty(self):
        return self.size == 0

    def enqueue(self, timestamp, item):
        current_timestamp = timestamp
        self._discard_old_entries(current_timestamp)
        
        if self.is_full():
            # Overwrite the oldest element
            self.head = (self.head + 1) % self.capacity
        else:
            self.size += 1
        
        # Insert the element into the buffer
        self.buffer[self.tail] = item
        self.timestamps[self.tail] = current_timestamp
        self.tail = (self.tail + 1) % self.capacity # Increase the tail index

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty buffer")
        
        # Get the element and clean it from the buffer 
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.timestamps[self.head] = None
        self.size -= 1
        self.head = (self.head + 1) % self.capacity # Increase the head index
        
        return item

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty buffer")
        return self.buffer[self.head]

    def get_valid_items(self):
        current_time = self.timestamps[self.head]
        valid_items = []
        index = self.head
        for _ in range(self.size):
            if current_time - self.timestamps[index] <= self.max_period:
                valid_items.append(self.buffer[index])
            index = (index + 1) % self.capacity
        return valid_items
    
    def get_period(self):
        if self.size < 2:
            period = 0
        else:
            period = self.timestamps[self.tail-1] - self.timestamps[self.head] 
        return period

    def _discard_old_entries(self, current_time):
        while not self.is_empty() and current_time - self.timestamps[self.head] > self.max_period:
            self.max_period_flag = True
            self.dequeue()

    def __repr__(self):
        return f"CircularBuffer({self.buffer}, head={self.head}, tail={self.tail}, size={self.size})"

#######################################################################################

# Example usage
if __name__ == "__main__":
    buffer = CircularBuffer(100, 5)  # 3-capacity buffer, 
                                     # discards items older than 5 seconds
    buffer.enqueue(1,1)
    buffer.enqueue(3,2)
    buffer.enqueue(5,3)
    buffer.enqueue(8,4)
    buffer.enqueue(9,5)
    buffer.enqueue(10,9)

    print(buffer.get_valid_items())

