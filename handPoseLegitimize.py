import pickle

import torch.nn as nn
import torch
import trimesh
from utils import *
from config import *

class HandPoseLegitimizeLayer(nn.Module):
    def __init__(self,fingerPlanarize=True,flexionLegitimize=False,abductionLegitimize=True,
                 planeRotationLegitimize=True,debug=False,r=9,relaxplane=False,useRHDangle=False):
        super(HandPoseLegitimizeLayer, self).__init__()
        ##only works for right hand!!!
        self.fingerPlanarize=fingerPlanarize
        self.flexionLegitimize=flexionLegitimize
        self.abductionLegitimize=abductionLegitimize
        self.planeRotationLegitimize=planeRotationLegitimize
        self.debug=debug
        self.relaxplane = relaxplane
        self.r=r
        self.useRHDangle=useRHDangle
        print("old---fingerPlanarize,flexionLegitimize,abductionLegitimize,planeRotationLegitimize,r,relaxplane",
              fingerPlanarize,flexionLegitimize,abductionLegitimize,planeRotationLegitimize,r,relaxplane)

    def forward(self,joints:torch.Tensor):
        if(self.fingerPlanarize):
            joints=HandPoseLegitimizeLayer.FingerPlanarize(joints,self.relaxplane)
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
        #project four finger point onto one average plane
        #the average plane can be estimated by section:5.4 Planarization in the paper or average over several three point plane
        # projection method is on the link https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
        # input: estimated joints
        # output: estimated joints after finger Planarization
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

    def PlaneRotationLegitimize(self,joints:torch.Tensor)->torch.Tensor:
        # twist rectication
        # paper 5.6 Twist Rectification
        # only works on each finger's mcp
        # input: estimated joints
        # output: estimated joints after finger Twist Rectification


        N = joints.shape[0]
        angleN = torch.tensor([np.pi / self.r],device=joints.device, dtype=joints.dtype)
        njoints = joints.clone()
        childern = [[2, 3, 17], [3, 17],
                    [5, 6, 18], [6, 18],
                    [11, 12, 19],[12, 19],
                    [8, 9, 20],[9, 20],
                    [14, 15, 16],[14, 15],
                    ]

        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20]]  # 1
        for idx, finger in enumerate(jidx):
            for yaxis in range(1, 2):

                mcppip,pipdip,maskmcppipdipline,overextensionmask,stdfingerPlaneDir,palmNorm,calculatedFingerDir=\
                    self.getoverextensionMask(joints,mcpidx=finger[1])

                angle = torch.acos(torch.clamp(torch.sum(calculatedFingerDir * stdfingerPlaneDir, dim=1), -1 + epsilon, 1 - epsilon)).reshape(-1)
                angle[maskmcppipdipline]*=0

                if(self.useRHDangle):
                    rot, difangle = self.getrot(twangle[finger[yaxis]], angle, 1, mcppip, calculatedFingerDir, stdfingerPlaneDir)
                else:
                    rot,difangle=self.getrot(angleN,angle,1,mcppip,calculatedFingerDir,stdfingerPlaneDir)

                ch=childern[idx*2+yaxis-1]

                for child in ch[1:]:
                    # rotate the whole finger.
                    t1 = (njoints[:, child] - njoints[:, ch[0]]).reshape(N, 3, 1)
                    njoints[:, child] = (rot @ t1).reshape(N, 3) + njoints[:, ch[0]]
        return njoints

    def getoverextensionMask(self,joints: torch.Tensor,mcpidx):
        # get useful vectors and for hyper-extension mask (finger which finger is in hyper-extension statue)
        fingeridx={1:[1,2,3],4:[4,5,6],10:[10,11,12],7:[7,8,9],13:[13,14,15]}
        mcp2normidx={1:0,4:1,10:2,7:3,13:4}
        mcp2stdfingeridx={1:0,4:1,10:2,7:3,13:4}
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
        # getrot: rotate in two directions, check which direction is better
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
        # Abduction and Adduction Rectification
        # paper section 5.5
        # only works on each finger's mcp
        # input: estimated joints
        # output: estimated joints after finger Abduction and Adduction Rectificatio
        device=joints.device
        N = joints.shape[0]
        normidx = [0, 1, 2, 3, 4]  # index,middle,ringy,pinky,thumb
        mcpidx = [1, 4, 10, 7,13]
        pipidx = [2, 5, 11, 8,14]
        angleP = torch.tensor([np.pi / self.r, np.pi / self.r, np.pi / self.r, np.pi / self.r, np.pi / self.r],
                              device=joints.device, dtype=joints.dtype)
        rectify=torch.tensor([0.1890, 0.1331, -0.1491,0.0347,0],device=joints.device, dtype=joints.dtype)
        njoints = joints.clone()
        childern = [[2, 3, 17],[5, 6, 18],[11, 12, 19],[8, 9, 20],[14, 15, 16]]
        for i in range(len(normidx)):
            palmNorm = getPalmNormByIndex(joints, normidx[i]).reshape(N, 3)  # palm up
            # palm up vector, palm plane normal vector
            vh = palmNorm.reshape(N, 3)
            mcp = joints[:, mcpidx[i]].reshape(N, 3)
            vd = -torch.sum(mcp * vh, dim=1).reshape(N, 1)
            pip = joints[:, pipidx[i]].reshape(N, 3)
            wrist = joints[:, 0]
            projpip = projectPoint2Plane(pip, vh, vd)[1].reshape(N, 3)
            # project points to a plane
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

            if(self.useRHDangle):
                rot, difangle = self.getrot(abangle[mcpidx[i]], angle, flexRatio, palmNorm, mcpprojpip, wristmcp)
            else:
                rot,difangle=self.getrot(angleP[i],angle,flexRatio,palmNorm,mcpprojpip,wristmcp)

            for child in childern[i]:
                #apply to the whole finger
                t1 = (njoints[:, child] - njoints[:, mcpidx[i]]).reshape(N, 3, 1)
                njoints[:, child] = (rot @ t1).reshape(N, 3) + njoints[:, mcpidx[i]]
        return njoints


    @staticmethod
    def FlexionLegitimizeForSingleJoint(njoints:torch.Tensor,fidx,finger,i,j,useRHDangle=False):
        # Flexion and Extension Rectification
        # paper sec 5.7
        # works single joint
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
            angleP = torch.tensor([np.pi / 2, np.pi*3 / 4, np.pi / 2], device=njoints.device,
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
        # finger plane normal vector

        angle = torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1 + epsilon, 1 - epsilon)).reshape(N)

        assert torch.sum(angle < 0) == 0
        #removed = (torch.abs(torch.sum(a * b, dim=1)) > 0.95)
        #angle[removed] = 0

        # stdFingerNorm:standard right direction
        # fingernorm: finger plane normal vector
        sign = torch.sum(fingernorm * stdFingerNorm, dim=1).reshape(N)
        maskP = (sign >= 0)
        maskN = (sign < 0)
        rot = torch.eye(3).reshape(1, 3, 3).repeat(N, 1, 1).reshape(N, 3, 3).to(device)
        if (torch.sum(maskP)):
            #equation: on (5.27)
            if(useRHDangle):
                difangle = torch.max(angle[maskP] - flexangle[rotroot[fidx * 3 + i]], torch.zeros_like(angle[maskP]))
            else:
                difangle = torch.max(angle[maskP] - angleP[i], torch.zeros_like(angle[maskP]))
            rot[maskP] = rotation_matrix(axis=fingerrotnorm[maskP], theta=-difangle)
        if (torch.sum(maskN)):
            # equation: on (5.27)
            if (useRHDangle):
                difangle = torch.max(angle[maskN] - extenangle[rotroot[fidx * 3 + i]], torch.zeros_like(angle[maskN]))
            else:
                difangle = torch.max(angle[maskN] - angleN[i], torch.zeros_like(angle[maskN]))
            rot[maskN] = rotation_matrix(axis=fingerrotnorm[maskN], theta=-difangle)

        idx = fidx * 3 + i
        for child in childern[idx]:
            t1 = (njoints[:, child] - njoints[:, rotroot[idx]]).reshape(N, 3, 1)
            njoints[:, child] = (rot @ t1).reshape(N, 3) + njoints[:, rotroot[idx]]
        return rot

    def FlexionLegitimize(self,joints: torch.Tensor) -> torch.Tensor:
        # Flexion and Extension Rectification
        # paper sec 5.7
        # works one each joint
        N = joints.shape[0]
        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18],  [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
        fidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0,1, 2], ]
        normidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0,1, 2], ]
        njoints = joints.clone()

        for fidx, finger in enumerate(jidx):
            #if (fidx == 4): angleN = angleNthumb
            for i,j in zip(fidces[fidx],normidces[fidx]):
                HandPoseLegitimizeLayer.\
                    FlexionLegitimizeForSingleJoint(njoints,fidx,finger,i,j,self.useRHDangle)
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


