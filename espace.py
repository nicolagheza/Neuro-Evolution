import pygame
import time
import random

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
FPS = 60
SCREENWIDTH = 288.0
SCREENHEIGHT = 512.0

# image and hitmask dicts
IMAGES, HITMASKS = {}, {}

load_saved_pool = 1
save_current_pool = 1
current_pool = []
fitness = []
total_models = 10

generation = 1

def save_pool():
    for xi in range(total_models):
        current_pool[xi].save_weights("Current_Model_Pool/first_model" + str(xi) + ".keras")
    print("Saved current pool!")

def model_crossover(model_idx1, model_idx2):
    global current_pool
    weights1 = current_pool[model_idx1].get_weights()
    weights2 = current_pool[model_idx2].get_weights()
    weightsnew1 = weights1
    weightsnew2 = weights2
    weightsnew1[0] = weights2[0]
    weightsnew2[0] = weights1[0]
    return np.asarray([weightsnew1, weightsnew2])

def model_mutate(weights):
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if random.uniform(0, 1) > 0.85:
                change = random.uniform(-0.5,0.5)
                weights[xi][yi] += change
    return weights

def predict_action(agentX, rocketX, rocketY, model_num):
    global current_pool
    agentX = min(SCREENWIDTH, agentX) / SCREENWIDTH - 0.5
    rocketX = min(SCREENWIDTH, rocketX) / SCREENWIDTH - 0.5
    rocketY = min(SCREENHEIGHT, rocketY) / SCREENHEIGHT - 0.5
    neural_input = np.asarray([agentX, rocketX, rocketY])
    neural_input = np.atleast_2d(neural_input)
    output_prob = current_pool[model_num].predict(neural_input, 1)[0]
    if np.argmax(output_prob) == 0:
        return 0
    return 1
       
    
# Initialize all models
for i in range(total_models):
    model = Sequential()
    model.add(Dense(input_dim=3, units=7))
    model.add(Activation("sigmoid"))
    model.add(Dense(units=2))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])
    current_pool.append(model)
    fitness.append(-100)

if load_saved_pool:
    for i in range(total_models):
        current_pool[i].load_weights("Current_Model_Pool/first_model"+str(i)+".keras")
def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Evolutionary Spaceship 0.1')

    IMAGES['background'] = pygame.image.load('img/background.png')
    IMAGES['agent'] = pygame.image.load('img/good_spaceship.png')
    IMAGES['rocket'] = pygame.image.load('img/rocket.png')

    #hitmask for player
    HITMASKS['agent'] = getHitmask(IMAGES['agent'])
    HITMASKS['rocket'] = getHitmask(IMAGES['rocket'])
    while True:
        gameloop()
        showGameOverScreen()
    
def gameloop():
    global fitness
    score = 0
    # Players 
    playersState = [] # True (alive) or False (dead)
    playersPosition = []
    playersVelX = []
    spaceship_acceleration = 5
    
    for idx in range(total_models):
        playersPosition.append([(SCREENWIDTH * 0.45), (SCREENHEIGHT * 0.80)])
        playersVelX.append(0)
        playersState.append(True)

    # Rocket
    rocket = getRandomRocket()
    rocketVelY = 4 

    alive_players = total_models
    
    while True:                              
        for event in pygame.event.get(): # QUIT THE GAME
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # Keyboard 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    playersVelX[0] = -spaceship_acceleration
                if event.key == pygame.K_RIGHT:
                    playersVelX[0] = spaceship_acceleration
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    playersVelX[0] = 0

        # check for crash here, return status list
        crashTest = checkCrash(playersPosition, rocket)

        for idx in range(total_models):
            if playersState[idx] == True and crashTest[idx] == True:
                alive_players -= 1
                playersState[idx] = False
            if playersState[idx] == True:
                fitness[idx] += 1
                action = predict_action(playersPosition[idx][0], rocket[0], rocket[1], idx)
                if action == 0:
                    playersVelX[idx] = -spaceship_acceleration
                elif action == 1:
                    playersVelX[idx] = spaceship_acceleration
                
                nextPlayerPosition = (playersPosition[idx][0] + playersVelX[idx])
                if  nextPlayerPosition + IMAGES['agent'].get_width() > SCREENWIDTH or nextPlayerPosition < 0: 
                    continue
                playersPosition[idx][0] += playersVelX[idx]
                    
        if alive_players == 0:
            return True
        
        # move rocket 
        rocket[1] += rocketVelY

        # check for scores
        for idx in range(total_models):
            if playersState[idx] == True and rocket[1] > SCREENHEIGHT:
                score += 1
                fitness[idx] += 25
                
        # get new rocket when it leaves the screen 
        if rocket[1] > SCREENHEIGHT:
            rocket = getRandomRocket()
            
        # draw background
        SCREEN.blit(IMAGES['background'], (0,0))

        # draw rocket
        SCREEN.blit(IMAGES['rocket'], (rocket[0],rocket[1]))

        # print score so player overlaps the score
        showScore(score)
        
        # draw player(s) 
        for idx, player in enumerate(playersPosition):
            if playersState[idx] == True:
                SCREEN.blit(IMAGES['agent'], player)
        
        pygame.display.update()
        FPSCLOCK.tick(FPS)

def getRandomRocket():
    """returns a randomly generated rocket"""
    rocketX = random.randrange(0, SCREENWIDTH)
    rocketY = -512
    return [rocketX, rocketY]

def showScore(score):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Dodged: " + str(score), True, (255,255,255))
    SCREEN.blit(text,(0,0))

    
def checkCrash(players, rocket):
    """ return True if player crash with rocket """
    statuses = []
    for idx in range(total_models):
        statuses.append(False)
        
    for idx in range(total_models):
        playerW = IMAGES['agent'].get_width()
        playerH = IMAGES['agent'].get_height()
        playerRect = pygame.Rect(players[idx][0], players[idx][1],
                        playerW, playerH)
        rocketW = IMAGES['rocket'].get_width()
        rocketH = IMAGES['rocket'].get_height()
        rocketRect = pygame.Rect(rocket[0], rocket[1], rocketW, rocketH)

        rHitMask = HITMASKS['rocket']
        pHitMask = HITMASKS['agent']

        # agent <--> rocket collision
        rCollide = pixelCollision(playerRect, rocketRect, pHitMask, rHitMask)

        if rCollide:
            statuses[idx] = True
            
    return statuses

def showGameOverScreen():
    """Perform genetic updates here"""
    global current_pool
    global fitness
    global generation
    new_weights = []
    total_fitness = 0
    for select in range(total_models):
        total_fitness += fitness[select]
    print ("total_fitness:", total_fitness)
    for select in range(total_models):
        fitness[select] /= total_fitness
        if select > 0:
            fitness[select] += fitness[select-1]
    for select in range(int(total_models/2)):
        parent1 = random.uniform(0,1)
        parent2 = random.uniform(0,1)
        idx1 = -1
        idx2 = -1
        for idxx in range(total_models):
            if fitness[idxx] >= parent1:
                idx1 = idxx
                break
        for idxx in range(total_models):
            if fitness[idxx] >= parent2:
                idx2 = idxx
                break
        new_weights1 = model_crossover(idx1, idx2)
        updated_weights1 = model_mutate(new_weights1[0])
        updated_weights2 = model_mutate(new_weights1[1])
        new_weights.append(updated_weights1)
        new_weights.append(updated_weights2)
    for select in range(len(new_weights)):
        fitness[select] = -100
        current_pool[select].set_weights(new_weights[select])
    if save_current_pool == 1:
        save_pool()
    generation = generation + 1
    return

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
    
