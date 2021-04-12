import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# create default 'linesafe' colourmap for matplotlib
color_array = plt.get_cmap('viridis')(range(256))
color_array[:,-1] = np.linspace(0.0,1.0,256)
map_object = LinearSegmentedColormap.from_list(name='linesafe',colors=color_array)
plt.register_cmap(cmap=map_object)

def update_figure_unet(X, y, prediction):
    plt.clf()

    ax1 = plt.subplot(1,3,1)
    ax1.imshow(X.view(-1,256), cmap='gray')
    
    ax2 = plt.subplot(1,3,2)
    ax2.imshow(X.view(-1,256), cmap='gray')
    ax2.imshow(y.view(-1,256), cmap='linesafe', alpha=0.6)

    ax3 = plt.subplot(1,3,3)
    ax3.imshow(X.view(-1,256), cmap='gray')
    ax3.imshow(prediction.view(-1,256), cmap='linesafe', alpha=0.6)

    plt.pause(0.05)

def prevent_figure_close():
    plt.show()
    
if __name__ == '__main__':
    pass




    
    
    