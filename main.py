import torch.nn as nn
import torch
from handPoseLegitimize import *
from utils import *

class KinematicLayer(nn.Module):
    def __init__(self, hpl=None,flex=True,notip=False,debug=False,datasetname=None):
        super(KinematicLayer, self).__init__()
        print("mano use flex rectification",flex,notip,datasetname)
        self.flex=flex
        if(hpl is None):
            self.hpl = HandPoseLegitimizeLayer(fingerPlanarize=True,flexionLegitimize=False,abductionLegitimize=True,
                 planeRotationLegitimize=True,debug=False,r=9,relaxplane=False)
        else:self.hpl = hpl
        self.notip=notip
        self.debug=debug
        self.datasetname=datasetname
        with open(manoPath, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)


    def matchTemplate2JointsGreedyWithConstraint(self,joint_gt:np.ndarray,tempJ=None):

        N = joint_gt.shape[0]
        joint_gt = joint_gt.reshape(N, 21, 3)
        if (not torch.is_tensor(joint_gt)):
            joint_gt = torch.tensor(joint_gt, device='cpu', dtype=torch.float32)
        device = joint_gt.device


        # first make wrist to zero

        orijoint_gt=joint_gt.clone()
        oriWrist = orijoint_gt[:, 0:1, :].clone()
        joint_gt = joint_gt- oriWrist.clone()

        transformG = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformL = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformLmano = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformG[:, 0, :3, 3] = joint_gt[:, 0].clone()
        transformL[:, 0, :3, 3] = joint_gt[:, 0].clone()
        transformLmano[:, 0, :3, 3] = joint_gt[:, 0].clone()



        if(tempJ is None):
            assert False
        else:
            #print("use external template")
            if(not torch.is_tensor(tempJ)):tempJ=torch.tensor(tempJ,dtype=torch.float32,device=device)
            if(len(tempJ.shape)==3):
                tempJ=tempJ.reshape(N, 21, 3)
            else:
                tempJ = tempJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3)
        tempJori = tempJ.clone()
        tempJ = tempJ - tempJori[:, 0:1, :]
        tempJori = tempJori - tempJori[:, 0:1, :].clone()




        R,t = wristRotTorch(tempJ, joint_gt)
        transformG[:, 0, :3, :3] = R
        transformG[:, 0, 3:, :3] = t
        transformL[:, 0, :3, :3] = R
        transformL[:, 0, 3:, :3] = t
        transformLmano[:, 0, :3, :3] = R
        transformLmano[:, 0, 3:, :3] = t


        joint_gt = self.hpl(joint_gt)

        #print(joint_gt,tempJ)
        assert (torch.sum(joint_gt[:,0]-tempJ[:,0])<1e-5),"wrist joint should be same!"+str(torch.sum(joint_gt[:,0]-tempJ[:,0]))

        childern = [[1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16],
                    [2, 3, 17], [3, 17], [17],
                    [5, 6, 18], [6, 18], [18],
                    [8, 9, 20], [9, 20], [20],
                    [11, 12, 19], [12, 19], [19],
                    [14, 15, 16], [15, 16], [16]]

        for child in childern[0]:
            t1 = (tempJ[:,child] - tempJ[:,0]).reshape(N,3,1)
            tempJ[:,child] = (transformL[:,0].clone() @ getHomo3D(t1)).reshape(N,4,1)[:,:-1,0]


        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        jidx = [[0], [1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
        mcpidx=[1,4,10,7]
        ratio = []
        for idx, i in enumerate(manoidx):
            pi = manopdx[idx]
            v0 = (tempJ[:,i] - tempJ[:,pi]).reshape(N,3)
            v1 = (joint_gt[:,i] - tempJ[:,pi]).reshape(N,3)

            # print('ratio',pi,i,torch.mean(torch.norm(v0)/torch.norm(joint_gt[:,i]-joint_gt[:,pi])))
            # ratio.append(np.linalg.norm(v0) / np.linalg.norm(v1))

            tr = torch.eye(4, dtype=torch.float32,device=device).reshape(1, 4, 4).repeat(N,1,1)
            r = getRotationBetweenTwoVector(v0, v1)
            tr[:,:3, :3] = r.clone()
            t0 = (tempJ[:,pi]).reshape(N,3)
            tr[:,:-1, -1] = t0

            # print('r',r)

            transformL[:,idx + 1] = tr


            for child in childern[pi]:
                t1 = (tempJ[:,child] - tempJ[:,pi]).reshape(N,3,1)
                tempJ[:,child] = (transformL[:,idx + 1].clone() @ getHomo3D(t1)).reshape(N,4,1)[:,:-1,0]

            jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
            if self.datasetname=='STB':
                fidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2], ]
                normidces = [[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 2], ]
            else:
                fidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],]
                normidces = [[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2], [0, 1, 2],]
            if(self.flex):
                if(self.notip and idx%3==2):pass
                elif(idx//3==4 and idx%3==2):pass
                else:
                    pass
                    rot=self.hpl.FlexionLegitimizeForSingleJoint(tempJ,fidx=idx//3,finger=jidx[idx//3],i=fidces[idx//3][idx%3],j=fidces[idx//3][idx%3],debug=self.debug)
                    tr[:,:3, :3] = rot@r.clone()
                    #print('rot',rot)
            transformL[:, idx + 1] = tr

            Gp = transformG[:, self.parents[idx + 1]].reshape(N, 4, 4).clone()
            transformG[:, idx + 1] = transformL[:, idx + 1].clone() @ Gp
            transformLmano[:, idx + 1] = torch.inverse(Gp) @ transformL[:, idx + 1].clone() @ Gp


        local_trans = transformLmano[:, 1:, :3, :3].reshape(N, 15, 3, 3)
        wrist_trans = transformLmano[:, 0, :3, :3].reshape(N, 1, 3, 3)

        outjoints = rotate2joint(wrist_trans, local_trans, tempJori, self.parents).reshape(N,21,3)
        assert (torch.mean(torch.sqrt(torch.sum((outjoints-tempJ)**2,dim=2)))<2),"outjoints and tempJ epe should be small"+str(torch.mean(torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))))

        outjoints = outjoints + oriWrist

        #return wrist_trans,local_trans,outjoints
        return outjoints



if __name__ == '__main__':
    pass
    joints=torch.randn(5,21,3)
    ref=getRefJoints(joints)
    kl=KinematicLayer()
    outjoints=kl.matchTemplate2JointsGreedyWithConstraint(joints,ref)
    print(outjoints.shape)

