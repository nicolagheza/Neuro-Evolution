import pygame
import time
import random

FPS = 60
SCREENWIDTH = 288.0
SCREENHEIGHT = 512.0

# image and hitmask dicts
IMAGES, HITMASKS = {}, {}

total_models = 1

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
    gameloop()
    
def gameloop():
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
                nextPlayerPosition = (playersPosition[idx][0] + playersVelX[idx])
                if  nextPlayerPosition + IMAGES['agent'].get_width() > SCREENWIDTH or nextPlayerPosition < 0: 
                    continue
                playersPosition[idx][0] += playersVelX[idx]
                
                    
        if alive_players == 0:
            return {}
        
        # move rocket 
        rocket[1] += rocketVelY

        # check for scores
        for idx in range(total_models):
            if playersState[idx] == True and rocket[1] > SCREENHEIGHT:
                score += 1
                
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
    
