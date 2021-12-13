import torch.nn as nn
import torch
from utils import *

class HandPoseLegitimizeLayer(nn.Module):
    def __init__(self):
        super(HandPoseLegitimizeLayer, self).__init__()

    def forward(self,joints:torch.Tensor,arr:torch.Tensor):
        self.AbductionLegitimize(joints,arr=arr)
        self.PlaneRotationLegitimize(joints,arr=arr)
        self.FlexionLegitimize(joints,arr=arr)
        #ab->tw->flex
        return joints


    def PlaneRotationLegitimize(self,joints:torch.Tensor,arr:torch.Tensor):
        mcpidx = [1, 4, 10, 7,13]

        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20]]  # 1
        for idx, finger in enumerate(jidx):
            for yaxis in range(1, 2):

                mcppip,pipdip,maskmcppipdipline,overextensionmask,stdfingerPlaneDir,palmNorm,calculatedFingerDir=\
                    self.getoverextensionMask(joints,mcpidx=finger[1])

                angle = torch.acos(torch.clamp(torch.sum(calculatedFingerDir * stdfingerPlaneDir, dim=1), -1 + epsilon, 1 - epsilon)).reshape(-1)
                angle[maskmcppipdipline]*=0

                arr[mcpidx[idx]][1] = angle[0] / 3.14 * 180

    def getoverextensionMask(self,joints: torch.Tensor,mcpidx):
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
    def AbductionLegitimize(self,joints: torch.Tensor,arr:torch.Tensor=None):
        device=joints.device
        N = joints.shape[0]
        normidx = [0, 1, 2, 3]  # 6
        normidx = [0, 1, 2, 3,4]  #7 index,middle,ringy,pinky,thumb
        normidx = [0, 1, 2, 3, 4]  # 1

        mcpidx = [1, 4, 10, 7,13]
        pipidx = [2, 5, 11, 8,14]
        fingerStdDiridx=[0,1,2,3,4]
        rectify=torch.tensor([0.1890, 0.1331, -0.1491,0.0347,0],device=joints.device, dtype=joints.dtype)
        njoints = joints.clone()
        childern = [[2, 3, 17],[5, 6, 18],[11, 12, 19],[8, 9, 20],[14, 15, 16]]
        for i in range(len(normidx)):
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

            arr[mcpidx[i]][0]=angle[0] / 3.14 * 180


    @staticmethod
    def FlexionLegitimizeForSingleJoint(njoints:torch.Tensor,fidx,finger,i,j):
        N,device=njoints.shape[0],njoints.device
        stdFingerNorm = getFingerStdDir(njoints, fidx)

        a0, a1, a2 = njoints[:, finger[i]], njoints[:, finger[i + 1]], njoints[:, finger[i + 2]]
        a, b = unit_vector(a1 - a0), unit_vector(a2 - a1)
        N = a.shape[0]
        fingernorm = unit_vector(torch.cross(a, b, dim=1))


        angle = torch.acos(torch.clamp(torch.sum(a * b, dim=1), -1 + epsilon, 1 - epsilon)).reshape(N)

        assert torch.sum(angle < 0) == 0
        sign = torch.sum(fingernorm * stdFingerNorm, dim=1).reshape(N)
        maskP = (sign >= 0)
        maskN = (sign < 0)
        if (torch.sum(maskN)):
            angle=-angle
        return angle

    def FlexionLegitimize(self,joints: torch.Tensor,arr: torch.Tensor):
        N = joints.shape[0]
        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18],  [0, 7, 8, 9, 20], [0, 10, 11, 12, 19], [0, 13, 14, 15, 16]]
        fidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0,1, 2], ]
        normidces = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0,1, 2], ]
        njoints = joints.clone()

        for fidx, finger in enumerate(jidx):
            for i,j in zip(fidces[fidx],normidces[fidx]):
                angle=HandPoseLegitimizeLayer.\
                    FlexionLegitimizeForSingleJoint(njoints,fidx,finger,i,j)
                # print(finger[i+1])
                if(float(angle[0])>0):
                    arr[finger[i+1]][2]=angle[0] / 3.14 * 180
                else:
                    arr[finger[i + 1]][3] = -angle[0] / 3.14 * 180



if __name__ == "__main__":
    import statistic.RHDDiscreteDataloader as RHDDiscreteDataloader
    train_dataset = RHDDiscreteDataloader.RHDDuscreteDataloader(train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1,shuffle=False)
    import tqdm
    arr=torch.zeros([len(train_dataset),21,4])
    print(arr.shape)
    hpl = HandPoseLegitimizeLayer()
    for idx, dic in tqdm.tqdm(enumerate(train_loader)):
        #if(idx>10):break
        xyz = dic["kp_coord_xyz"]
        hpl(xyz,arr[idx])
        mcpidx=[1,4,7,10,13]
        flexidx=[1,2,3,  4,5,6, 7,8,9, 10,11,12, 13,14,15]
        print(idx,"ab",arr[idx][mcpidx][:,0])
        print(idx,"tw",arr[idx][mcpidx][:,1])
        print(idx,"flex",arr[idx][flexidx][:,2])
        print(idx,"exten",arr[idx][flexidx][:,3])
    arr=arr.cpu().numpy()
    print('ab',torch.tensor(np.percentile(arr[:,:,0],75,axis=0)))
    print('tw',torch.tensor(np.percentile(arr[:,:,1],75,axis=0)))
    print('flex',torch.tensor(np.percentile(arr[:,:,2],75,axis=0)))
    print('exten',torch.tensor(np.percentile(arr[:,:,3],75,axis=0)))
    print('ab',torch.tensor(np.max(arr[:,:,0],axis=0)))
    print('tw',torch.tensor(np.max(arr[:,:,1],axis=0)))
    print('flex',torch.tensor(np.max(arr[:,:,2],axis=0)))
    print('exten',torch.tensor(np.max(arr[:,:,3],axis=0)))
