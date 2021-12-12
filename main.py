import torch.nn as nn
import torch
from handPoseLegitimize import *
from utils import *
from config import *

class KinematicLayer(nn.Module):
    def __init__(self,RHDtemplate=False,flex=True,useRHDAngle=False):
        super(KinematicLayer, self).__init__()
        self.flex=flex
        self.hpl = HandPoseLegitimizeLayer(fingerPlanarize=True, flexionLegitimize=False, abductionLegitimize=True,
                                           planeRotationLegitimize=True, debug=False, r=9, relaxplane=False,useRHDangle=useRHDAngle)
        self.parents = np.array([-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14]).astype(np.int32)
        self.RHDtemplate=RHDtemplate
        self.useRHDAngle=useRHDAngle
        if RHDtemplate:
            self.bonelenstd=boneLenStd
            self.bonelenmean=boneLenMean


    def AlignStretchTemplateWithConstraint(self,joints,tempJ):
        assert torch.is_tensor(joints) and torch.is_tensor(tempJ)
        N,device = joints.shape[0],joints.device

        joints = joints.reshape(N, 21, 3)
        wrist = joints[:, 0:1, :].clone()
        joints = joints.clone() - wrist.clone()
        tempJ=tempJ.clone()-tempJ[:,:1,:].clone()
        bonelenTempJ=tempJ.clone()
        R, t = wristRotTorch(tempJ, joints)

        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12, 13, 14, 15]
        transformL = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(N, 1, 1)
        transformL[:, :3, :3] = R
        transformL[:, 3:, :3] = t
        for child in [1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16]:
            t1 = (tempJ[:,child] - tempJ[:,0]).reshape(N,3,1)
            tempJ[:,child] = (transformL.clone() @ getHomo3D(t1)).reshape(N,4,1)[:,:-1,0]
        plamsIdx = [0, 1, 4, 7, 10]

        joints[:, plamsIdx] = tempJ[:, plamsIdx].clone()
        joints = self.hpl(joints)

        for idx, i in enumerate(manoidx):
            pi = manopdx[idx]
            v1 = (joints[:,i] - tempJ[:,pi]).reshape(N,3)
            dis=torch.norm(v1,dim=1).reshape(N)
            univ1=v1/dis.reshape(N,1)
            if(self.RHDtemplate):
                upperbound=(self.bonelenmean[i]+self.bonelenstd[i]).reshape(1).repeat(N).reshape(N)
                lowerbound=(self.bonelenmean[i]-self.bonelenstd[i]).reshape(1).repeat(N).reshape(N)
            else:
                upperbound = torch.norm(bonelenTempJ[:, i] - bonelenTempJ[:, pi],dim=1).reshape(N)
                lowerbound = torch.norm(bonelenTempJ[:, i] - bonelenTempJ[:, pi],dim=1).reshape(N)
            validmask=((lowerbound<=dis)&(dis<=upperbound))
            lessmask=dis<lowerbound
            moremask=dis>upperbound
            if(torch.sum(validmask)):tempJ[validmask,i]=joints[validmask,i].clone()
            if(torch.sum(lessmask)):tempJ[lessmask,i]=tempJ[lessmask,pi]+(univ1*lowerbound.reshape(N,1))[lessmask]
            if(torch.sum(moremask)):tempJ[moremask,i]=tempJ[moremask,pi]+(univ1*upperbound.reshape(N,1))[moremask]
            jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
            fidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],]
            if(self.flex):
                rot=self.hpl.FlexionLegitimizeForSingleJoint(tempJ,fidx=idx//3,finger=jidx[idx//3],i=fidces[idx//3][idx%3],j=fidces[idx//3][idx%3],useRHDangle=self.useRHDAngle)
                t1 = (tempJ[:,i] - tempJ[:,pi]).reshape(N,3,1)
                tempJ[:,i] = (rot @ t1).reshape(N,3) + tempJ[:,pi].reshape(N,3)

        return wrist+tempJ



if __name__ == '__main__':
    joints=torch.randn(5,21,3)
    ref=getRefJoints(joints)
    kl=KinematicLayer(RHDtemplate=True,useRHDAngle=True)
    outjoints=kl.AlignStretchTemplateWithConstraint(joints,ref)
    print(outjoints.shape)



    from cscPy.mano.network.manolayer import MANO_SMPL
    from cscPy.mano.network.utils import *
    from cscPy.Const.const import manoPath
    mano_right = MANO_SMPL(manoPath, ncomps=45, oriorder=True,
                           device='cuda', userotJoints=True)
    skeleton2skinepe=[]
    for epoch in range(1, 100):
        pose = torch.tensor(np.random.uniform(-2, 2, [7, 45]).astype(np.float32))
        rootr = torch.tensor(np.random.uniform(-3.14, 3.14, [7, 3]).astype(np.float32))
        vertex_gt, joint_gt = mano_right.get_mano_vertices(rootr.view(7, 1, 3),
                                                           pose.view(7, 45),
                                                           torch.zeros([70]).view(7, 10),
                                                           torch.ones([7]).view(7, 1),
                                                           torch.zeros([21]).view(7, 3),
                                                           pose_type='pca', mmcp_center=False)

        joint_gt = mano_right.newjs.cpu().numpy()[:, :, :-1].copy().reshape(7, 21, 3)
        joint_gt = get32fTensor(joint_gt)

        templatejoints = getRefJoints(joint_gt)
        # print(joint_gt.shape,templatejoints.shape)

        tempJ = kl.AlignStretchTemplateWithConstraint(get32fTensor(joint_gt),tempJ=get32fTensor(templatejoints))
        tempJ = tempJ.cpu().numpy().copy()
        joint_gt = joint_gt.cpu().numpy().copy()

        print("epe tempJ,joint_gt", np.mean(np.sqrt(np.sum((tempJ - joint_gt) ** 2, axis=-1))) * 1000)
        skeleton2skinepe.append(np.mean(np.sqrt(np.sum((tempJ - joint_gt) ** 2, axis=-1))) * 1000)

    print("mean skeleton2skinepe", np.mean(skeleton2skinepe))


