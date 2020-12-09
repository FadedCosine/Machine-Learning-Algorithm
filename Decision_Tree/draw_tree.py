import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
from Data_Loader.car_data_loader import datalabel as feature_names, id2feature as feature_x_names
little_down = 0.03
little_left = 0.03
def getNumLeafs(TreeRoot):
    numLeafs = 0
    if len(TreeRoot.child) == 0:
        return numLeafs + 1
    else:
        for child in TreeRoot.child:
            numLeafs += getNumLeafs(child)    
        return numLeafs

def getTreeDepth(TreeRoot):
    maxDepth = 0
    if len(TreeRoot.child) == 0:
        return 1
    else:
        for child in TreeRoot.child:
            maxDepth = max(maxDepth, getNumLeafs(child) + 1)    
        return maxDepth

def plotMidText(txtString, cntrPt, parentPt): # 在箭头中间的文字
    xMid = (parentPt[0] - little_left - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - little_down - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, fontsize=8)

def plotNode(nodeTxt, centerPt, parentPt, nodeType): #这是箭头，以及箭头最终的结点，以及节点中的文字
    little_down_parentPt = (parentPt[0], parentPt[1]-little_down)
    createPlot.ax1.annotate(nodeTxt, xy=little_down_parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotTree(TreeRoot, parentPt, arrowStr, kind="NotCART"):
    numLeafs = getNumLeafs(TreeRoot)
    depth = getTreeDepth(TreeRoot)

    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalw, plotTree.yOff)
    
    suffix = ":{}".format(feature_x_names[TreeRoot.split_feature][TreeRoot.cart_split_feature_x]) if TreeRoot.cart_split_feature_x is not None else ""
    plotNode(feature_names[TreeRoot.split_feature] + suffix, cntrPt, parentPt, decisionNode)
    plotMidText(arrowStr, cntrPt, parentPt)
  
    plotTree.yOff = plotTree.yOff - 1.8 / plotTree.totalD
    for child in TreeRoot.child:
        if len(child.child) == 0:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalw
            if kind != "CART":
                # print(feature_x_names[TreeRoot.split_feature][child.pre_split_feature_x])
                plotMidText(feature_x_names[TreeRoot.split_feature][child.pre_split_feature_x], (plotTree.xOff, plotTree.yOff), cntrPt)
            else:
                plotMidText("yes" if child.pre_split_feature_x==0 else "no", (plotTree.xOff, plotTree.yOff), cntrPt)
                # plotNode("yes" if child.pre_split_feature_x else "no", (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotNode(feature_x_names[-1][child.y],  (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            
        else:
            if kind != "CART":
                plotTree(child, cntrPt, feature_x_names[TreeRoot.split_feature][child.pre_split_feature_x])
            else:
                plotTree(child, cntrPt, "yes" if child.pre_split_feature_x==0 else "no", kind="CART")
    plotTree.yOff = plotTree.yOff + 1.8 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
#ID3-Tree
def ID3_Tree(inTree, file_name):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    # plt.title("ID3-Tree",fontsize=12,color='red')
    plt.savefig(file_name)
    plt.show()

#C4.5-Tree
def C45_Tree(inTree, file_name):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    # plt.title("C4.5-Tree",fontsize=12,color='red')
    plt.savefig(file_name)
    plt.show()

#CART-Tree
def CART_Tree(inTree, file_name):
    fig = plt.figure(3, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '', kind="CART")
    # plt.title("CART-Tree",fontsize=12,color='red')
    plt.savefig(file_name)
    plt.show()
