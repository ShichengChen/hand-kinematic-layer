import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

epsilon=1e-6
import os
manoPath='/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
if not os.path.exists(manoPath):
    manoPath = '/home/shicheng/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'
if not os.path.exists(manoPath):
    manoPath = '/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl'

def get32fTensor(a)->torch.Tensor:
    if(torch.is_tensor(a)):
        return a.float()
    return torch.tensor(a,dtype=torch.float32)

def getRefJoints(joint_gt):
    #get rest position of the hand
    manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    if(torch.is_tensor(joint_gt)):
        N=joint_gt.shape[0]
        OriManotempJ = torch.tensor(joint_gt.reshape(N, 21, 3),dtype=torch.float32,device=joint_gt.device)
        manotempJ = OriManotempJ.clone()
        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = torch.norm(OriManotempJ[:, ci] - OriManotempJ[:, pi], dim=-1, keepdim=True)+1e-8
            dm = torch.norm(OriManotempJ[:, pi] - OriManotempJ[:, ppi], dim=-1, keepdim=True)+1e-8
            manotempJ[:, ci] = manotempJ[:, pi] + (manotempJ[:, pi] - manotempJ[:, ppi]) / dm * dp
        return manotempJ
    else:
        OriManotempJ = joint_gt.reshape(21, 3)
        manotempJ = OriManotempJ.copy()
        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = np.linalg.norm(OriManotempJ[ci] - OriManotempJ[pi]) + 1e-8
            dm = np.linalg.norm(OriManotempJ[pi] - OriManotempJ[ppi]) + 1e-8
            manotempJ[ci] = manotempJ[pi] + (manotempJ[pi] - manotempJ[ppi]) / dm * dp
        return manotempJ

def wristRotTorch(tempJ,joint_gt):
    #palm registration
    #get rotation R and transition t
    #paper 5.3.1 Palm Registration
    #svd rigid alignment https://zhuanlan.zhihu.com/p/115135931
    assert torch.is_tensor(tempJ)
    plamsIdx=[0,1,4,7,10]
    N, n, d = tempJ.shape[0], len(plamsIdx), tempJ.shape[2]
    a=tempJ[:,plamsIdx].clone().reshape(N,n,3)
    b=joint_gt[:,plamsIdx].clone().reshape(N,n,3)
    aave = torch.sum(a, dim=1, keepdim=True) / n
    bave = torch.sum(b, dim=1, keepdim=True) / n
    x = (a - aave.reshape(N,1,d)).reshape(N, n, d)
    y = (b - bave.reshape(N,1,d)).reshape(N, n, d)
    r = svdForRotationWithoutW(x, y)
    t = bave - aave @ r
    return r,t.reshape(N,1,d)


def getPalmNorm(joints: torch.Tensor, ) -> torch.Tensor:
    #palm plane normal vector
    palmNorm = unit_vector(torch.cross(joints[:, 4] - joints[:, 0], joints[:, 7] - joints[:, 4], dim=1))
    return palmNorm

def getPalmNormByIndex(joints: torch.Tensor, idx: int) -> torch.Tensor:
    #get one palm plane normal vector
    if (idx == -1): return getPalmNorm(joints)
    assert 0 <= idx <= 4, "bad index"
    #c = [(13, 1), (1, 4), (4, 10), (10, 7)] (18/9/21 version) 1223normidx
    c = [(1, 4), (4, 10), (4, 10), (10, 7), (13, 1)]
    #c = [(1, 10), (1, 10), (1, 10), (10, 7), (13, 1)]
    #c = [(1, 10), (1, 10), (4, 7), (4, 7),(13, 1),] #new
    return unit_vector(
        torch.cross(joints[:, c[idx][0]] - joints[:, 0], joints[:, c[idx][1]] - joints[:, c[idx][0]], dim=1))


def rotation_matrix(axis:torch.Tensor, theta:torch.Tensor)->torch.Tensor:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    N=max(theta.shape[0],axis.shape[0])
    assert len(axis.shape)==2
    axis = unit_vector(axis).reshape(N,3)
    a = torch.cos(theta / 2.0).reshape(N)
    curbcd = -axis * torch.sin(theta / 2.0).reshape(N,1)
    b, c, d = curbcd[:,0],curbcd[:,1],curbcd[:,2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    out=torch.zeros([N,3,3],dtype=theta.dtype,device=theta.device)
    out[:,0,0]=aa + bb - cc - dd
    out[:,0,1]=2 * (bc + ad)
    out[:,0,2]=2 * (bd - ac)
    out[:,1,0]=2 * (bc - ad)
    out[:,1,1]=aa + cc - bb - dd
    out[:,1,2]=2 * (cd + ab)
    out[:,2,0]=2 * (bd + ac)
    out[:,2,1]=2 * (cd - ab)
    out[:,2,2]=aa + dd - bb - cc
    return out


def getFingerStdDir(joints:torch.Tensor,idx:int)->torch.Tensor:
    #get standard right vector
    # paper equation (5.2)
    #N = joints.shape[0]
    #if(idx==0):return unit_vector(joints[:,1]-joints[:,4])
    # if(0<=idx<=2):return unit_vector(joints[:,1]-joints[:,10])
    # elif(idx==3):return unit_vector(joints[:,4]-joints[:,7])
    # elif idx==4:return unit_vector(joints[:,13]-joints[:,1])

    if (idx <= 1):
        return unit_vector(joints[:, 1] - joints[:, 10])
    if (3 >= idx >= 2):
        return unit_vector(joints[:, 4] - joints[:, 7])
    elif idx == 4:
        return unit_vector(joints[:, 13] - joints[:, 1])
    # normidx = [-1, -1, -1, -1, 0]  # index,middle,ringy,pinky,thumb
    # mcpidx = [1, 4, 10, 7, 13]
    # palmNorm = getPalmNormByIndex(joints, normidx[idx]).reshape(N, 3)  # palm up
    # vecWristMcp = unit_vector(joints[:, mcpidx[idx]] - joints[:, 0]).reshape(N, 3)  # wirst to mmcp
    # stdFingerNorm = unit_vector(torch.cross(vecWristMcp, palmNorm, dim=1))  # direction from pmcp to imcp
    # return stdFingerNorm

def euDist(v0,v1):
    return torch.sqrt(epsilon*1e-3+torch.sum((v0-v1)**2,dim=-1))


def disPoint2Plane(points,planeNorm,planeD):
    #distance of plane to point
    #https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    N = points.shape[0]
    return (torch.sum((points.reshape(N,3) * planeNorm.reshape(N,3)).reshape(N, 3), dim=1, keepdim=True)
           + planeD.reshape(N,1)).reshape(N, 1)

def projectPoint2Plane(points,planeNorm,planeD,relaxValue=0):
    # https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    # project point to a plane
    N=points.shape[0]
    points=points.reshape(N,3)
    planeNorm=unit_vector(planeNorm)
    #https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    dis=disPoint2Plane(points,planeNorm,planeD).reshape(N,1)
    #plane: ax+by+cz+d=0
    #norm: (a,b,c)
    #dis=norm*point+d
    if(relaxValue==0):
        pass
    else:
        mask=dis<0
        dis=torch.max(torch.abs(dis) - relaxValue, torch.zeros_like(dis))
        dis[mask]*=-1
    projectedPoint = (points - dis*planeNorm.reshape(N,3)).reshape(N, 3)
    #ans=point-dist*norm
    return torch.abs(epsilon+dis),projectedPoint

def getPlaneFrom4Points(joints: torch.Tensor)->(torch.Tensor,torch.Tensor):
    # get a average finger plane from four joints of one finger.
    N,n,d=joints.shape
    assert n==4
    from itertools import combinations
    subsets = list(combinations([0,1,2,3], 3))
    subsets = [[0,2,3],[0,1,3]]
    #subsets = [[0,1,2],[0,1,3]]
    vlist = []
    for subset in subsets:
        v0 = joints[:, subset[0]] - joints[:, subset[1]]
        v1 = joints[:, subset[1]] - joints[:, subset[2]]
        vh = torch.cross(v0, v1, dim=1)
        vlist.append(vh.reshape(1, N, 3))
    vh = unit_vector(torch.mean(torch.cat(vlist, dim=0), dim=0).reshape(N, 3)).reshape(N,1,3)
    #vh = torch.mean(torch.cat(vlist, dim=0), dim=0).reshape(N, 1,3)
    subj = joints.reshape(N, 4, 3)
    vd = torch.mean(-torch.sum(subj * vh, dim=2), dim=1).reshape(N, 1)
    return vh,vd

def svdForRotationWithoutW(a,b):
    #svd rigid-registration
    #https://zhuanlan.zhihu.com/p/115135931
    N,n,d=a.shape[0],a.shape[1],a.shape[2]
    w = (a.permute(0, 2, 1)) @ b
    w = w.cpu()
    u, _, v = w.svd()
    s = v.reshape(N, d, d) @ (u.reshape(N, d, d).permute(0, 2, 1))
    s = torch.det(s)
    #print(N,n,d)
    I = torch.eye(d,d).reshape(1,d,d).repeat(N,1,1).float()
    I[:,-1,-1]=s.clone()
    #print(I.shape)
    #s = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, s]])
    r = v.reshape(N, d, d) @ I @ (u.reshape(N, d, d).permute(0, 2, 1))
    r = r.to(a.device).reshape(N,3,3)
    return r

def unit_vector(vec):
    if(torch.is_tensor(vec)):
        bs=vec.shape[0]
        vec=vec.reshape(bs,3)
        return vec / (torch.norm(vec,dim=1,keepdim=True)+1e-8)
    return vec / (np.linalg.norm(vec)+1e-8)
def getRotationBetweenTwoVector(a,b):
    #https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    #paper 5.3.2 Finger Registration
    if(torch.is_tensor(a)):
        #print('a,b',a,b)
        device=a.device
        bs = a.shape[0]
        a = unit_vector(a)
        b = unit_vector(b)
        a=a.reshape(bs,3)
        G=torch.eye(3,dtype=torch.float32,device=device).reshape(1,3,3).repeat(bs,1,1).reshape(bs,3,3)
        G[:, 0, 0] = torch.sum(a * b, dim=1)
        G[:, 0, 1] = -torch.norm(torch.cross(a,b,dim=1), dim=1)
        G[:, 1, 0] = torch.norm(torch.cross(a,b,dim=1), dim=1)
        G[:, 1, 1] = torch.sum(a * b, dim=1)
        u=a.clone()
        v=b-torch.sum(a*b,dim=1,keepdim=True)*a
        v = unit_vector(v)
        F = torch.zeros([bs,3,3],dtype=torch.float32,device=device)
        F[:,:, 0], F[:,:, 1], F[:,:, 2] = u, v, unit_vector(torch.cross(b, a, dim=1))

        f = F.cpu()
        #print('f',f)
        #print(np.linalg.matrix_rank(f))
        rf = (torch.sum(torch.svd(f)[1]>1e-4,dim=1) == 3)
        if(rf.device!=device):rf=rf.to(device)
        R = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3).repeat(bs, 1, 1).reshape(bs, 3, 3)
        R[rf] = F[rf] @ G[rf] @ torch.inverse(F[rf])
        return R
    else:
        a=unit_vector(a).copy()
        b=unit_vector(b).copy()
        if (np.linalg.norm(a - b) < 1e-5): return np.eye(3)
        G=np.array([[np.dot(a,b),- np.linalg.norm(np.cross(a,b)),0],[ np.linalg.norm(np.cross(a,b)),np.dot(a,b),0],[0,0,1]])
        u=a.copy()
        v=b-(np.dot(a,b))*a
        v=unit_vector(v)
        F=np.zeros([3,3],dtype=np.float64)
        F[:,0],F[:,1],F[:,2]=u,v,unit_vector(np.cross(b,a))
        R=F@G@np.linalg.inv(F)
        return R



def getHomo3D(x):
    # get homogeneous vector
    # [x,y,z]->[x,y,z,1]
    if(torch.is_tensor(x)):
        if(x.shape[-1]==4):return x
        if(x.shape[-1]==1 and x.shape[-2]==4):return x
        if(x.shape[-1]==1 and x.shape[-2]==3):
            return torch.cat([x, torch.ones([*(x.shape[:-2])] + [1,1], dtype=torch.float32,device=x.device)], dim=-2)
        if(x.shape[-1]==3):
            return torch.cat([x, torch.ones([*(x.shape[:-1])] + [1], dtype=torch.float32,device=x.device)], dim=-1)
    if(x.shape[-1]==3):
        return np.concatenate([x,np.ones([*(x.shape[:-1])]+[1],dtype=np.float64,device=x.device)],axis=-1)
    return x


def rotate2joint(wrist_trans,local_trans,template,parent):
    #input: rotation matrix for each joints and template
    #output: rotated hand skeleton
    device = wrist_trans.device
    Rs = torch.cat([wrist_trans, local_trans], dim=1)
    N = Rs.shape[0]
    root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(template, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        ones_homo = Variable(torch.ones(N, 1, 1))
        ones_homo = ones_homo.to(device)
        t_homo = torch.cat([t, ones_homo], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    newjs = template.clone()

    newjsones = torch.ones([newjs.shape[0], 21, 1]).to(device)
    newjs = torch.cat([newjs, newjsones], dim=2).reshape(N, 21, 4, 1)
    orijs = newjs.clone()
    transidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    cpidx = [1, 4, 7, 10, 13]

    for i in range(5):
        a = minusHomoVectors(orijs[:, cpidx[i]], orijs[:, 0]).reshape(N,4,1)
        newjs[:, cpidx[i]] = (A0 @ a)

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)

        a = minusHomoVectors(orijs[:, transidx[i - 1]], orijs[:, i])
        newjs[:, transidx[i - 1]] = (res_here @ a).reshape(N,4,1)
        results.append(res_here)

    return newjs[:,:,:-1].reshape(N,21,3)


def minusHomoVectors(v0, v1):
    v = v0 - v1
    if (v.shape[-1] == 1):
        v[..., -1, 0] = 1
    else:
        v[..., -1] = 1
    return v


