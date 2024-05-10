import numpy as np
from skimage.util import view_as_windows as viewW
import h5py

def runIm(I):
	np.seterr(divide='ignore',invalid='ignore')
	global res
	prm={
		'minContrast':9,
		'removeEpsilon':0.248,
		'nmsFact':0.75,
		'maxTurn':35,
		'maxNumOfEdges':50,
		'complexity':15,
		'w':2,
		'sigma':0.1,
		'patchSize':5,
	}
	res={'prm':prm}
	n,m=I.shape
	with h5py.File('Mat/1.mat','r') as f:
		bot={key:f['bot'][key][()] for key in f['bot'].keys()}
	S=np.ravel_multi_index(np.meshgrid(range(n),range(n)),(n,n)).reshape(n,n)
	N=n**2
	res['N']=N
	for field in ['R','L','C','Ang0','Ang1','minC','maxC']:
		res[field]=np.full((N,N),np.nan,np.float64)
	res['S0']=np.full((N,N),-1,np.int32)
	res['pixels']=np.full((N,N,prm['patchSize']),-1,np.int32)
	QPFast(I,S,bot,N,prm)
	min_test=((res['C']>0)&(res['minC']>=(res['C']/2)))|((res['C']<0)&(res['maxC']<=(res['C']/2)))
	res['C'][~min_test]=np.nan
	getScores(prm)
	res_out=res['E'].copy()
	res.clear()
	return res_out

def im2col(A,BSZ,stepsize=1):
	return viewW(A.T,(BSZ[0],BSZ[1])).reshape(-1,BSZ[0]*BSZ[1]).T[:,::stepsize]

def QPFast(I,S,bot,N,prm):
	global res
	n=int(np.sqrt(N))
	w=prm['w']
	step=prm['patchSize']-1
	Ipad=np.pad(I,((w,w),(w,w)),'symmetric')
	Ipatches=im2col(Ipad,(2*w+step+1,2*w+step+1))
	Spatches=im2col(S,(step+1,step+1))
	xInd=np.arange(np.floor(step/2)+1,n-np.floor(step/2)+1,step,int)
	xInd=(xInd-np.floor(step/2)).astype(int)-1
	Ind=np.ravel_multi_index(np.meshgrid(xInd,xInd),(n-step,n-step)).flatten()
	Ipatches=Ipatches[:,Ind]
	Spatches=Spatches[:,Ind]
	getBottomLevelFast(Ipatches,Spatches,bot,prm)
	maxJ=int(np.log2(n-1))-1
	for j in range(2,maxJ+1):
		m=2**j+1
		file_name=f'Mat/{m}.mat'
		with h5py.File(file_name,'r') as f:
			for it in ['tableSingle','tableDouble']:
				mergeSquaresFast(f[it][:].T,prm)

def mergeSquaresFast(table,prm):
	global res
	N=res['L'].shape[0]
	table=table.astype(int)-1
	s0=table[:,0]
	ind0s0,s0ind1,ind01,ind10=[(table[:,i+1]%N,table[:,i+1]//N) for i in range(4)]
	validLen=(res['L'][ind0s0]>=0)&(res['L'][s0ind1]>=0)
	len=res['L'][ind0s0]+res['L'][s0ind1]
	resp=res['R'][ind0s0]+res['R'][s0ind1]
	stitchAng=np.mod(res['Ang1'][ind0s0]-res['Ang0'][s0ind1],360)
	validAng=(stitchAng<=prm['maxTurn'])|(360-stitchAng<=prm['maxTurn'])
	minC=np.fmin(res['minC'][ind0s0],res['minC'][s0ind1])
	maxC=np.fmax(res['maxC'][ind0s0],res['maxC'][s0ind1])
	ang0=res['Ang0'][ind0s0]
	ang1=res['Ang1'][s0ind1]
	con=resp/len/prm['w']/2
	con[~validLen]=np.nan
	resp[~validLen]=np.nan
	minLen=(len<=prm['minContrast'])
	minC[minLen]=con[minLen]
	maxC[minLen]=con[minLen]
	newRes=(np.abs(con)>np.abs(res['C'][ind01]))|np.isnan(res['C'][ind01])
	valid=(validLen>0)&newRes&validAng
	data=np.vstack((con,len,resp,minC,maxC,table[:,3],table[:,4],s0,ang0,ang1)).T
	data=data[valid,:]
	scores=np.abs(data[:,0])-threshold(data[:,1])
	idx=np.argsort(scores)
	con,len,resp,minC,maxC,ind01,ind10,s0,ang0,ang1=[data[idx,i] for i in range(10)]
	li=[ind01,ind10]
	ind01,ind10=[(it.astype(int)%N,it.astype(int)//N) for it in li]
	res['R'][ind01]=resp
	res['R'][ind10]=-resp
	res['L'][ind01]=len
	res['L'][ind10]=len
	res['C'][ind01]=con
	res['C'][ind10]=-con
	res['Ang0'][ind01]=ang0
	res['Ang1'][ind01]=ang1
	res['Ang0'][ind10]=np.mod(ang1+180,360)
	res['Ang1'][ind10]=np.mod(ang0+180,360)
	res['minC'][ind01]=minC
	res['minC'][ind10]=-maxC
	res['maxC'][ind01]=maxC
	res['maxC'][ind10]=-minC
	res['S0'][ind01]=s0
	res['S0'][ind10]=s0

def getBottomLevelFast(Ipatches,Spatches,bot,prm):
	global res
	resp=np.dot((bot['leftVec']-bot['rightVec']),Ipatches)
	ind0,ind1=[Spatches[bot[it].astype(int)-1,:].reshape(150,256) for it in ['p0','p1']]
	angle=getAngle(ind0,ind1,int(np.sqrt(res['N'])))
	indices=bot['indices'].reshape(-1).astype(int)-1
	isZero=(indices==-1)
	indices[isZero]=0
	lineIndices=Spatches[indices,:]
	lineIndices[isZero,:]=-1
	len=np.tile(bot['lengthVec'],(1,Spatches.shape[1]))
	rSize=res['L'].shape
	ind01=(ind0,ind1)
	ind10=(ind1,ind0)
	N=res['L'].shape[0]
	for cord in range(prm['patchSize']):
		curCord=lineIndices[cord::prm['patchSize'],:]
		temp=np.full((N,N),-1)
		temp[ind01]=curCord
		temp[ind10]=curCord
		res['pixels'][:,:,cord]=temp
	con=resp/len/prm['w']/2
	con[(len==0)|np.isnan(len)]=np.nan
	bad=np.abs(con)<(prm['removeEpsilon']*prm['sigma'])
	con[bad]=np.nan
	len[bad]=np.nan
	resp[bad]=np.nan
	res['R'][ind01]=resp
	res['R'][ind10]=-resp
	res['L'][ind01]=len
	res['L'][ind10]=len
	res['C'][ind01]=con
	res['C'][ind10]=-con
	res['Ang0'][ind01]=angle
	res['Ang1'][ind01]=angle
	angle2=np.mod(angle+180,360)
	res['Ang0'][ind10]=angle2
	res['Ang1'][ind10]=angle2

def getScores(prm):
	global res
	N=res['L'].shape[0]
	n=int(np.sqrt(N))
	res['E']=np.zeros((n,n))
	selected=np.zeros((n,n),bool)
	SC=np.abs(res['C'])-threshold(res['L'])
	IN=np.tril(np.ones(SC.shape))
	SC=np.tril(SC)
	SC[IN==0]=np.nan
	Ind=np.arange((N**2))
	SC=SC.flatten()
	edge=(SC>0)&(~np.isnan(SC))
	SC=SC[edge]
	Ind=Ind[edge]
	idx=np.argsort(SC)[::-1]
	SC=SC[idx]
	Ind=Ind[idx]
	counter=0
	for i in range(len(SC)):
		ind0,ind1=np.unravel_index(Ind[i],(N,N))
		E=addEdge(ind0,ind1,np.zeros((n,n),bool),1)
		if E is None:
			continue
		Edialate=E|np.pad(E,((1,0),(0,0)))[:-1,:]|np.pad(E,((0,1),(0,0)))[1:,:]|np.pad(E,((0,0),(1,0)))[:,:-1]|np.pad(E,((0,0),(0,1)))[:,1:]
		if np.sum(Edialate&selected)/np.sum(E)<prm['nmsFact']:
			counter+=1
			selected|=Edialate
			res['E']=np.fmax(res['E'],E*SC[i])
			if counter>prm['maxNumOfEdges']:
				return

def addEdge(ind0,ind1,E,level):
	global res
	if level==50:
		print('deep')
		return None
	s0=res['S0'][ind0,ind1]
	res['S0'][ind0,ind1]=-1
	s1=-1
	if s1==ind0 or s1==ind1:
		s1=np.nan
	if s0==ind0 or s0==ind1:
		s0=-1
	if s1!=-1 and s0!=-1:
		E=addEdge(ind0,s0,E,level+1)
		if E is None:
			return None
		E=addEdge(s0,s1,E,level+1)
		if E is None:
			return None
		E=addEdge(s1,ind1,E,level+1)
		if E is None:
			return None
	elif s0!=-1:
		E=addEdge(ind0,s0,E,level+1)
		if E is None:
			return None
		E=addEdge(s0,ind1,E,level+1)
		if E is None:
			return None
	else:
		pixels=res['pixels'][ind0,ind1,:].flatten()
		pixels=pixels[pixels!=-1]
		E[(pixels%65,pixels//65)]=True
		if len(pixels)==0:
			return None
	return E

def threshold(L):
	global res
	prm=res['prm']
	alpha=4
	beta=prm['complexity']*2-1
	w=2*prm['w']
	T=prm['sigma']*np.sqrt(2*(np.log(6*res['N'])+0*(beta*L/alpha)*np.log(2))/(w*L))
	return T

def getAngle(ind0,ind1,n):
	y0,x0=np.unravel_index(ind0,(n,n))
	y1,x1=np.unravel_index(ind1,(n,n))
	v1=x1-x0
	v2=y1-y0
	angle=np.arctan(v2/v1)*180/np.pi
	angle[v1<0]+=180
	angle=np.mod(angle,360)
	return angle
