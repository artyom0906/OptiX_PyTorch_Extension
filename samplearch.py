class Card:
    def __init__(self, task_queue):
        self.geometry = []
        self.task_queue = task_queue
        pass

    def transformAction(self, transform):
        def callBack():
            self.geometry = transform(self.geometry)

        return lambda : self.task_queue.append(callBack)
def rotate(x, y):
    def transformation(geometry):
        for g in geometry:
            pass # x, y
    return transformation

if __name__ == '__main__':
    taskQueue = []
    card = Card(taskQueue)
    action = card.transformAction(rotate(180, 0))
    #....
    action()#add transform to queue in button