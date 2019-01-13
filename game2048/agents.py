import numpy as np
import math
from keras.models import load_model
import torch

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    #def data_save(self,max_iter=np.inf,verbose=False):
     #   n_iter=0
     #   while (n_iter < max_iter) and (not self.game.end):
     #       direction = self.step()
     #       self.game.move(direction)
     #       n_iter += 1


    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class aAgent(Agent):
    from mymodel import Net
    def step(self):
        self.m1 = self.m2 = self.m3 = self.m4 = np.zeros((4, 4))
        self.model= torch.load('model.pth')
        self.input=self.game.board.reshape(16,1)
        self.input = self.input.astype(int)
        for i in range(len(self.input)):
            if self.input[i] > 0:
                self.input[i] = math.log(self.input[i], 2)
        self.m1=np.reshape(self.input,(4,4))
        self.m2 = self.m1.T
        for j in range(4):
            self.m3[j] = self.m1[3-j]
        self.m4 = self.m3.T
        self.input1 = np.reshape(self.m1, (1,1,4,4))
        self.input1=torch.from_numpy(self.input1).float().cuda()
        self.predict1 = self.model(self.input1)
        self.predict1 = self.predict1.cuda().data.cpu().numpy().tolist()
        self.predict1=max(self.predict1)
        self.flag1=self.predict1.index(max(self.predict1))

        self.input2 = np.reshape(self.m2, (1,1,4,4))
        self.input2 = torch.from_numpy(self.input2).float().cuda()
        self.predict2 = self.model(self.input2)
        self.predict2 = self.predict2.cuda().data.cpu().numpy().tolist()
        self.predict2 = max(self.predict2)
        self.flag2 = self.predict2.index(max(self.predict2))
        if self.flag2==0:
            self.flag2=4
        self.flag2 = self.flag2-1


        self.input3 = np.reshape(self.m3, (1,1,4,4))
        self.input3 = torch.from_numpy(self.input3).float().cuda()
        self.predict3 = self.model(self.input3)
        self.predict3 = self.predict3.cuda().data.cpu().numpy().tolist()
        self.predict3 = max(self.predict3)
        self.flag3 = self.predict3.index(max(self.predict3))
        if self.flag3<2:
            self.flag3 = self.flag3 +4
        self.flag3=self.flag3-2

        self.input4 = np.reshape(self.m4, (1,1,4,4))
        self.input4 = torch.from_numpy(self.input4).float().cuda()
        self.predict4 = self.model(self.input4)
        self.predict4 = self.predict4.cuda().data.cpu().numpy().tolist()
        self.predict4 = max(self.predict4)
        self.flag4 = self.predict4.index(max(self.predict4))
        if self.flag4<3:
            self.flag4=self.flag4+4
        self.flag4 = self.flag4-3

        if self.flag1 == self.flag1 and \
                self.flag1 == self.flag3 :
            direction = self.flag1
        else:
            direction = self.flag4
        return direction

class MyAgent(Agent):
    def step(self):
        self.m1=self.m2=self.m3=self.m4=np.zeros((4,4))
        self.model=load_model('model_256.h5')
        self.input = self.game.board.reshape(1,16)
        self.input = self.input.astype(int)
        self.input = max(self.input)
        for i in range(len(self.input)):
            if self.input[i] > 0:
                self.input[i]=math.log(self.input[i], 2)
        self.input = [i + 1 for i in self.input]
        self.m1 = np.reshape(self.input, (1,4,4,1))
        self.m2 = self.m1.T
        for j in range(4):
            self.m3[j] = self.m1[3-j]
        self.m4 = self.m3.T
        self.input1 = np.concatenate((np.concatenate((self.m1, self.m2), axis=0), np.concatenate((self.m4, self.m3), axis=0)),axis=1).reshape(1,8,8,1)
        #left 90 input
        self.input2 = np.concatenate((np.concatenate((self.m2, self.m1), axis=0), np.concatenate((self.m3, self.m4), axis=0)),axis=1).reshape(1,8,8,1)
        #left 180 input
        self.input3 = np.concatenate((np.concatenate((self.m3, self.m4), axis=0), np.concatenate((self.m2, self.m1), axis=0)), axis=1).reshape(1,8,8,1)
        # left 180 input
        self.input4 = np.concatenate((np.concatenate((self.m4, self.m3), axis=0), np.concatenate((self.m1, self.m2), axis=0)), axis=1).reshape(1,8,8,1)
        self.predict1=np.reshape(self.input1, (1,8,8,1))
        self.predict1 = self.model.predict(self.input1)
        self.predict1 = max(self.predict1.tolist())
        self.direction1 = self.predict1.index(max(self.predict1))
        self.predict2=np.reshape(self.input2, (1, 8, 8, 1))
        self.predict2 = self.model.predict(self.input2)
        self.predict2 = max(self.predict2.tolist())
        self.direction2 = self.predict2.index(max(self.predict2))
        self.direction2 = (self.direction2+2)%3
        self.predict3=np.reshape(self.input3, (1, 8, 8, 1))
        self.predict3 = self.model.predict(self.input3)
        self.predict3 = max(self.predict3.tolist())
        self.direction3 = self.predict3.index(max(self.predict3))
        self.direction3 = (self.direction3+1)%3
        self.predict4=np.reshape(self.input4, (1, 8, 8, 1))
        self.predict4 = self.model.predict(self.input4)
        self.predict4 = max(self.predict4.tolist())
        self.direction4 = self.predict4.index(max(self.predict4))
        self.direction4 = self.direction4%3
        if self.predict1 == self.direction2 and \
            self.direction1==self.direction3 and\
            self.direction1==self.direction4:
            direction=self.direction1
        elif self.direction2==self.direction3 and \
             self.direction2==self.direction4:
             direction=self.direction2
        else:
            direction=self.direction3
        return direction

