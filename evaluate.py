from game2048.game import Game
from game2048.displays import Display
from mymodel import Net

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    from game2048.agents import aAgent as TestAgent1
    from game2048.agents import ExpectiMaxAgent as TestAgent2
    from game2048.agents import MyAgent as TestAgent3

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,AgentClass=TestAgent1)
        scores.append(score)
        print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
