import pickle

import torch.nn as nn
import torch
import trimesh
from utils import *


class HandPoseLegitimizeLayer(nn.Module):
    def __init__(self,fingerPlanarize=True,flexionLegitimize=False,abductionLegitimize=True,
                 planeRotationLegitimize=True,debug=False,r=9,relaxplane=False):
        super(HandPoseLegitimizeLayer, self).__init__()
        ##only works for right hand!!!
        self.fingerPlanarize=fingerPlanarize
        self.flexionLegitimize=flexionLegitimize
        self.abductionLegitimize=abductionLegitimize
        self.planeRotationLegitimize=planeRotationLegitimize
        self.debug=debug
        self.relaxplane = relaxplane
        self.r=r
        print("old---fingerPlanarize,flexionLegitimize,abductionLegitimize,planeRotationLegitimize,r,relaxplane",
              fingerPlanarize,flexionLegitimize,abductionLegitimize,planeRotationLegitimize,r,relaxplane)

    def forward(self,joints:torch.Tensor):
        if(self.fingerPlanarize):
            joints=HandPoseLegitimizeLayer.FingerPlanarize(joints,self.relaxplane)
            #visMeshfromJoints(joints)
        if (self.abductionLegitimize):
            joints = self.AbductionLegitimize(joints)
        if (self.planeRotationLegitimize):
            joints = self.PlaneRotationLegitimize(joints)
            joints = HandPoseLegitimizeLayer.FingerPlanarize(joints,self.relaxplane)
        if(self.flexionLegitimize):
            joints=self.FlexionLegitimize(joints)
        return joints

    @staticmethod
    def FingerPlanarize(joints: torch.Tensor,relaxplane=False) -> torch.Tensor:
        N = joints.shape[0]
        njoints = joints.clone()
        #norms = joints.clone()
        jidx = [[1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
        for finger in jidx:
            vh, vd = getPlaneFrom4Points(joints[:, finger].clone())
            for idx in range(4):
                #norms[:,finger[idx]]=vh.clone().reshape(N,3)
                if (relaxplane):
                    _, njoints[:, finger[idx]] = projectPoint2Plane(joints[:, finger[idx]], vh, vd,
                                                                    relaxValue=0.001)
                else:
                    _, njoints[:, finger[idx]] = projectPoint2Plane(joints[:, finger[idx]], vh, vd)
        return njoints

    def FingerBoneLengthLegitimize(self,joints: torch.Tensor, templateHand: torch.Tensor, order='mano') -> torch.Tensor:
        assert order == 'mano', "only support mano order"
        N, n, _ = joints.shape
        assert templateHand.shape == (N, 21, 3), "template have no batch"
        manopdx = [[2, 3, 17], [5, 6, 18], [8, 9, 20], [11, 12, 19], [14, 15, 16], ]
        manoppx = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], ]
        childern = [[2, 3, 17], [3, 17], [17],
                    [5, 6, 18], [6, 18], [18],
                    [8, 9, 20], [9, 20], [20],
                    [11, 12, 19], [12, 19], [19],
                    [14, 15, 16], [15, 16], [16]]
        njoints = joints.clone()
        for idx, (fu, fv) in enumerate(zip(manoppx, manopdx)):
            for i in range(3):
                cid = idx * 3 + i
                dt = euDist(templateHand[:, fv[i]], templateHand[:, fu[i]])
                ds = euDist(joints[:, fv[i]], joints[:, fu[i]])
                scale = dt / ds
                dirt = (joints[:, fv[i]] - joints[:, fu[i]]) * scale
                dirs = (joints[:, fv[i]] - joints[:, fu[i]])
                dif = dirt - dirs
                for j in childern[cid]:
                    njoints[:, j] = joints[:, j] + dif
        return njoints

    def PlaneRotationLegitimize(self,joints:torch.Tensor)->torch.Tensor:
        N = joints.shape[0]
        normidx = [1, 2, 2, 3]

        angleN = torch.tensor([np.pi / self.r],device=joints.device, dtype=joints.dtype)
        njoints = joints.clone()
        childern = [[2, 3, 17], [3, 17],
                    [5, 6, 18], [6, 18],
                    [11, 12, 19],[12, 19],
                    [8, 9, 20],[9, 20],
                    [14, 15, 16],[14, 15],
                    ]
        mcpidx = [1, 4, 10, 7]

        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20]]
        for idx, finger in enumerate(jidx):
            for yaxis in range(1, 2):

                mcppip,pipdip,maskmcppipdipline,overextensionmask,stdfingerPlaneDir,palmNorm,calculatedFingerDir=\
                    self.getoverextensionMask(joints,mcpidx=finger[1])

                angle = torch.acos(torch.clamp(torch.sum(calculatedFingerDir * stdfingerPlaneDir, dim=1), -1 + epsilon, 1 - epsilon)).reshape(-1)
                angle[maskmcppipdipline]*=0

                rot,difangle=self.getrot(angleN,angle,1,mcppip,calculatedFingerDir,stdfingerPlaneDir)

                ch=childern[idx*2+yaxis-1]
                #print('rot',ch, difangle)
                for child in ch[1:]:
                    t1 = (njoints[:, child] - njoints[:, ch[0]]).reshape(N, 3, 1)
                    njoints[:, child] = (rot @ t1).reshape(N, 3) + njoints[:, ch[0]]
                if (self.debug and torch.sum(difangle) > 1e-3 and N==1):
                    pass
                    print(idx, rot, difangle / 3.14 * 180, dir)
                    print('rot plane child', childern[idx*2+yaxis-1])
                    visMeshfromJoints(njoints)
        return njoints

    def getoverextensionMask(self,joints: torch.Tensor,mcpidx):
        fingeridx={1:[1,2,3],4:[4,5,6],10:[10,11,12],7:[7,8,9],}
        mcp2normidx={1:0,4:1,10:2,7:3}
        mcp2stdfingeridx={1:0,4:1,10:2,7:3}
        fi=fingeridx[mcpidx]
        N,device=joints.shape[0],joints.device
        wristmcp = unit_vector(joints[:, fi[0]] - joints[:, 0])
        mcppip = unit_vector(joints[:, fi[1]] - joints[:, fi[0]])
        pipdip = unit_vector(joints[:, fi[2]] - joints[:, fi[1]])
        maskmcppipdipline = (torch.abs(torch.sum(mcppip * pipdip, dim=1)) > 0.95)
        palmNorm = unit_vector(getPalmNormByIndex(joints, mcp2normidx[mcpidx]).reshape(N, 3)).to(device)  # palm up
        calculatedFingerDir = unit_vector(torch.cross(mcppip, pipdip, dim=1)).reshape(N, 3)
        calculatedFingerDir2 = unit_vector(torch.cross(wristmcp,mcppip, dim=1)).reshape(N, 3)
        stdfingerPlaneDir = getFingerStdDir(joints, mcp2stdfingeridx[mcpidx]).reshape(N, 3)
        overextensionmask = torch.sum(calculatedFingerDir * stdfingerPlaneDir, dim=1).reshape(N) < 0
        overextensionmask2 = torch.sum(calculatedFingerDir2 * stdfingerPlaneDir, dim=1).reshape(N) < 0
        return mcppip,pipdip,maskmcppipdipline,overextensionmask|overextensionmask2,\
               stdfingerPlaneDir,palmNorm,calculatedFingerDir

    def getrot(self,angleP,angle,flexRatio,rotaxis,startvec,endvec):
        N,device=angle.shape[0],angle.device
        rot = torch.eye(3).reshape(1, 3, 3).repeat(N, 1, 1).reshape(N, 3, 3).to(device)
        reversemask=angle>3.1415926/180*120
        angle[reversemask]=3.1415926-angle[reversemask]
        difangle = torch.max(angle - angleP, torch.zeros_like(angle)) * flexRatio
        rot0 = rotation_matrix(axis=rotaxis, theta=difangle)
        rot1 = rotation_matrix(axis=rotaxis, theta=-difangle)
        rot0mcpprojpip = unit_vector((rot0.reshape(N, 3, 3) @ startvec.reshape(N, 3, 1)).reshape(N, 3))
        rot1mcpprojpip = unit_vector((rot1.reshape(N, 3, 3) @ startvec.reshape(N, 3, 1)).reshape(N, 3))
        mask0 = torch.abs(torch.sum(rot0mcpprojpip * endvec, dim=1)) > torch.abs(torch.sum(rot1mcpprojpip * endvec, dim=1))
        mask1 = (~mask0)
        if (torch.sum(mask0)): rot[mask0] = rot0[mask0]
        if (torch.sum(mask1)): rot[mask1] = rot1[mask1]
        return rot,difangle
    def AbductionLegitimize(self,joints: torch.Tensor) -> torch.Tensor:
        device=joints.device
        N = joints.shape[0]
        normidx = [0, 1, 2, 3]  # index,middle,ringy,pinky,thumb
        mcpidx = [1, 4, 10, 7]
        pipidx = [2, 5, 11, 8]
        fingerStdDiridx=[0,1,2,3]
        #r = 18
        #r = 180
        angleP = torch.tensor([np.pi / self.r, np.pi / self.r, np.pi / self.r, np.pi / self.r],
                              device=joints.device, dtype=joints.dtype)
        rectify=torch.tensor([0.1890, 0.1331, -0.1491,0.0347],device=joints.device, dtype=joints.dtype)
        njoints = joints.clone()
        childern = [[2, 3, 17],[5, 6, 18],[11, 12, 19],[8, 9, 20],[14, 15, 16]]
        for i in range(4):
            palmNorm = getPalmNormByIndex(joints, normidx[i]).reshape(N, 3)  # palm up
            vh = palmNorm.reshape(N, 3)
            mcp = joints[:, mcpidx[i]].reshape(N, 3)
            vd = -torch.sum(mcp * vh, dim=1).reshape(N, 1)
            pip = joints[:, pipidx[i]].reshape(N, 3)
            wrist = joints[:, 0]
            projpip = projectPoint2Plane(pip, vh, vd)[1].reshape(N, 3)
            dis = euDist(mcp, pip).reshape(N)
            flexRatio = euDist(projpip, mcp).reshape(N) / (dis + epsilon)
            flexRatio[flexRatio < 0.3] = 0


            wristmcp = unit_vector(mcp - wrist).reshape(N, 3)
            mcpprojpip = unit_vector(projpip - mcp).reshape(N, 3)
            mcppip = unit_vector(pip - mcp).reshape(N, 3)
            overflexionmask = torch.acos(
                torch.clamp(torch.sum(wristmcp * mcppip, dim=1), -1 + epsilon, 1 - epsilon)).reshape(
                -1) > 3.14 / 2

            rectifiedwristmcp=(rotation_matrix(axis=palmNorm, theta=rectify[i:i+1].repeat(N))@wristmcp.reshape(N,3,1)).reshape(N,3)
            angle = torch.acos(torch.clamp(torch.sum(rectifiedwristmcp * mcpprojpip, dim=1), -1 + epsilon, 1 - epsilon)).reshape(-1)
            angle[overflexionmask]*=0
            rot,difangle=self.getrot(angleP[i],angle,flexRatio,palmNorm,mcpprojpip,wristmcp)

            for child in childern[i]:
                t1 = (njoints[:, child] - njoints[:, mcpidx[i]]).reshape(N, 3, 1)
                njoints[:, child] = (rot @ t1).reshape(N, 3) + njoints[:, mcpidx[i]]
            if (self.debug and torch.sum(difangle) > 1e-3 and N==1):
                pass
                print('pip projpip',pip, projpip)
                print(i, rot, difangle / 3.14 * 180,flexRatio)
                print('abdcution child', childern[i])
                visMeshfromJoints(njoints)
        return njoints


    @staticmethod
    def FlexionLegitimizeForSingleJoint(njoints:torch.Tensor,fidx,finger,i,j,debug=False):
        N,device=njoints.shape[0],njoints.device
        stdFingerNorm = getFingerStdDir(njoints, fidx)
        if (fidx <= 3):
            angleN = torch.tensor([np.pi / 4, np.pi / 18, np.pi / 4], device=njoints.device,
                                  dtype=njoints.dtype)  # .reshape(1, 3).repeat(N, 1)
            angleP = torch.tensor([np.pi / 2, np.pi * 3 / 4, np.pi / 2], device=njoints.device,
                                  dtype=njoints.dtype)  # .reshape(1, 3).repeat(N, 1)
        elif (fidx == 4):
            angleN = torch.tensor([np.pi / 4, np.pi / 18, np.pi / 4], device=njoints.device,
                                  dtype=njoints.dtype)  # .reshape(1, 3).repeat(N, 1)
            angleP = torch.tensor([np.pi / 4, np.pi / 4, np.pi / 2], device=njoints.device,
                                  dtype=njoints.dtype)  # .reshape(1, 3).repeat(N, 1)
        else:
            assert False

        childern = [[2, 3, 17], [3, 17], [17],
                    [5, 6, 18], [6, 18], [18],
                    [8, 9, 20], [9, 20], [20],
                    [11, 12, 19], [12, 19], [19],
                    [14, 15, 16], [15, 16], [16]]
        rotroot = [1,2,3 , 4,5,6 , 7,8,9 , 10,11,12, 13,14,15]
        a0, a1, a2 = njoints[:, finger[i]], njoints[:, finger[i + 1]], njoints[:, finger[i + 2]]
        a, b = unit_vector(a1 - a0), unit_vector(a2 - a1)
        N = a.shape[0]
        fingernorm = unit_vector(torch.cross(a, b, dim=1))

        a00, a11, a22 = njoints[:, finger[j]], njoints[:, finger[j + 1]], njoints[:, finger[j + 2]]
        fingerrotnorm=unit_vector(torch.cross(unit_vector(a11 - a00), unit_vector(a22 - a11),dim=1))


        angle = torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1 + epsilon, 1 - epsilon)).reshape(N)

        assert torch.sum(angle < 0) == 0
        #removed = (torch.abs(torch.sum(a * b, dim=1)) > 0.95)
        #angle[removed] = 0


        sign = torch.sum(fingernorm * stdFingerNorm, dim=1).reshape(N)
        maskP = (sign >= 0)
        maskN = (sign < 0)
        rot = torch.eye(3).reshape(1, 3, 3).repeat(N, 1, 1).reshape(N, 3, 3).to(device)
        difangle=0
        if (torch.sum(maskP)):
            difangle = torch.max(angle[maskP] - angleP[i], torch.zeros_like(angle[maskP]))
            rot[maskP] = rotation_matrix(axis=fingerrotnorm[maskP], theta=-difangle)
        if (torch.sum(maskN)):
            difangle = torch.max(angle[maskN] - angleN[i], torch.zeros_like(angle[maskN]))
            rot[maskN] = rotation_matrix(axis=fingerrotnorm[maskN], theta=-difangle)
            #if (i == 0): angleN[maskN][1:] *= 0.1

        idx = fidx * 3 + i
        for child in childern[idx]:
            t1 = (njoints[:, child] - njoints[:, rotroot[idx]]).reshape(N, 3, 1)
            njoints[:, child] = (rot @ t1).reshape(N, 3) + njoints[:, rotroot[idx]]
        if (debug and torch.sum(difangle) > 1e-3 and N==1):
            pass
            print(rotroot[idx], rot, angle / 3.14 * 180, difangle / 3.14 * 180, sign)
            print('flex child', childern[idx], angleN)
            visMeshfromJoints(njoints)
        return rot

    def FlexionLegitimize(self,joints: torch.Tensor) -> torch.Tensor:
        N = joints.shape[0]
        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18],  [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
        fidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0,1, 2], ]
        normidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0,1, 2], ]
        njoints = joints.clone()

        for fidx, finger in enumerate(jidx):
            #if (fidx == 4): angleN = angleNthumb
            for i,j in zip(fidces[fidx],normidces[fidx]):
                HandPoseLegitimizeLayer.\
                    FlexionLegitimizeForSingleJoint(njoints,fidx,finger,i,j,self.debug)
        return njoints



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def drawhandBone(joints, order='mano'):
    linecolor = ['green', 'magenta',  'cyan', 'yellow','white']
    assert order == 'mano'
    assert len(joints.shape) == 2
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal', adjustable='datalim')
    linesg = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
    for i in range(len(linesg)):
        ax.plot(joints[linesg[i], 0], joints[linesg[i], 1], joints[linesg[i], 2], marker='o', color=linecolor[i])

def visJoints(joints):
    joints = joints.detach().cpu().numpy().reshape(21, 3)
    drawhandBone(joints)

