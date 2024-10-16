import pygame
from sympy import false


def init():
    pygame.init()
    win = pygame.display.set_mode((600,600))

def getKey(keyName):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return  ans

def main():
    if getKey("LEFT"):
        print("Left Key pressed")
    if getKey("RIGHT"):
        print("Right Key pressed")

if __name__ == '__main__':
    init()
    while True:
        main()