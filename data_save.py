from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent
from itertools import chain
import operator
from functools import reduce
import numpy as np
import math
display1 = Display()
display2 = IPythonDisplay()
score = [];
data1=[];
data2=[];
data3=[];


# a random start
 #game = Game(4, random=False,score_to_win=32,enable_rewrite_board=True)
 #display2.display(game)
 #agent = ExpectiMaxAgent(game, display=display2)
 #agent.play(verbose=True)

# a random start
print("Running the loop manually...")

game = Game(4, random=False, enable_rewrite_board=False)
agent = ExpectiMaxAgent(game)
m=0;


while (m<1000):
    game = Game(4, random=False, enable_rewrite_board=False)
    agent = ExpectiMaxAgent(game)
    for _ in range(1000):
        direction = agent.step()
        #print("Moving to direction `%s`..."%direction)
        game.move(direction)
        #display1.display(game)
        array=game.board.flatten()
        for i in range(len(array)):
            if array[i] > 0:
                array[i]=math.log(array[i], 2)
        array = array.astype(int)
        array = np.append(array, direction)
        if np.max(array)<=8:
            data1 = np.append(data1, array)
        elif np.max(array) ==9:
            data2 = np.append(data2, array)
        elif np.max(array) ==10:
            data3 = np.append(data3, array)
        else:
            direction = agent.step()
            game.move(direction)
            display1.display(game)
            array = game.board.flatten()
            for i in range(len(array)):
                if array[i] > 0:
                    array[i] = math.log(array[i], 2)
            array = array.astype(int)
            array = np.append(array, direction)
            data3=np.append(data3,array)
            break;
    else:
        m=m+1;
        break;

        if self.flag1 == self.flag2 and \
                self.flag1 == self.flag3 and \
                self.flag1 == self.flag4:
            direction = self.flag1
        elif self.flag2 == self.flag3 and \
                self.flag2 == self.flag4:
            direction = self.flag2
        else:
            direction = self.flag3

np.save("256train_new.npy", data1)
np.save("512train_new.npy",data2)
np.save("1024train_new.npy", data3)
print(len(data1))
print(m)



