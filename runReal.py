import numpy as np
from runIm import runIm
def runReal(I):
	m,n=I.shape
	R=np.zeros_like(I)
	block=2**int(np.floor(np.log2(min(I.shape))))+1
	gap=10
	xGrid=getGrid(block,gap,m)
	yGrid=getGrid(block,gap,n)
	for x0 in xGrid:
		for y0 in yGrid:
			x0-=1
			y0-=1
			curI=I[x0:x0+block,y0:y0+block]
			curR=runIm(curI)
			curR[0:2,:]=0
			curR[-2:,:]=0
			curR[:,0:2]=0
			curR[:,-2:]=0
			R[x0:x0+block,y0:y0+block]=np.fmax(R[x0:x0+block,y0:y0+block],curR)
	return R

def getGrid(block,gap,maximum):
	grid=np.zeros(maximum,dtype=int)
	grid[0]=1
	for i in range(1,maximum):
		grid[i]=grid[i-1]+block-gap
		if grid[i]+block-1>maximum:
			grid[i]=maximum-block+1
			break
	grid=grid[grid!=0]
	if len(grid)>=2 and grid[-1]==grid[-2]:
		grid=grid[:-1]
	return grid
