import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('input', '/home/martin/Documents/HPCAssignment/Output', 'path to particle input files')
flags.DEFINE_integer('width', 10, 'width of the visualisation window in inches')
flags.DEFINE_integer('height', 10, 'height of the visualisation window in inches')
flags.DEFINE_float('framelength', 10, 'duration of each visualised frame')
flags.DEFINE_boolean('loop', True, 'animation looping')

def getParticles(lines):
    particles = []# Add each particle's values to this list for us to plot
    for line in lines:
        tokens = line.split(', ')
        particleVals = []
        for t in tokens:
            tokNum = float(t)
            particleVals.append(tokNum)
        particles.append(particleVals)
    return particles

def getValsFromFile(path):
    f = open(path, "r")
    lines = f.readlines()
    particles = getParticles(lines)
    masses = [p[0] for p in particles]
    xs = [p[1] for p in particles]
    ys = [p[2] for p in particles]
    return masses, xs, ys

def main(_argv):
    dirPath = FLAGS.input
    width = FLAGS.width
    height = FLAGS.height
    frameLength = FLAGS.framelength
    loop = FLAGS.loop

    numFrames = len([f for f in os.listdir(dirPath)])
    
    firstFile = dirPath + "/nbody_0.txt"# Get path of first file
    masses, xs, ys = getValsFromFile(firstFile)
   
    fig = plt.figure(figsize=(width, height))
    ax = plt.axes(xlim=(min(xs), max(xs)), ylim=(min(ys), max(ys)))
    ax.set_xticks([])
    ax.set_yticks([])

    scat = plt.scatter(xs, ys, s=masses, c="blue")
        
    def update(frame):
        nextFile = dirPath + "/nbody_" + str(frame) + ".txt"# Get path of next file
        newMasses, newXs, newYs = getValsFromFile(nextFile)
        
        positions = list(zip(newXs,newYs))

        scat.set_offsets(positions)
        scat.set_sizes(newMasses)
        ax.set_title(frame)
        return scat,

    anim = animation.FuncAnimation(fig, update, frames =numFrames, interval=frameLength, blit=False, repeat=loop)
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
