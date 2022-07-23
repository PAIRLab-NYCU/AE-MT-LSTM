# import 
import numpy as np
import glob

class route_generator:
    def __init__(self):
        self.coor = []
        for i in range(50):
            if i < 25:
                self.coor.append(np.array([i//5,4-i%5])) #[0,4][0,3][0,2][0,1][0,0][1,4]
            else:
                self.coor.append(np.array([(i-25)//5,(i-25)%5+5])) #[0,5][0,6][0,7][0,8][0,9][1,5]      
        
    def getCDFMap(self,p_location,d_t,m):
        ## calaulate normalized probability for every position and build CDF map
        ## inputs
        #  p_location: the previous location
        #  d_t: time interval
        v_max = np.random.normal(1.2,0.8)
        sigma = v_max * d_t
        co = self.coor
        if sigma==0:
            print('s=0')
        p = [np.exp(-(np.sum(np.square(t-p_location)))/(2*sigma**2)) for t in co]
        pm = [np.exp(-(np.sum(np.square(t+m-p_location)))/(2*sigma**2)) for t in co]
        p_norm = np.array(p)/sum(p)
        if sum(pm) == 0:
            for i in range(1,len(p_norm)):
                p_norm[i] += p_norm[i-1]
            return p_norm
        else:
            pm_norm = np.array(pm)/sum(pm)
            for i in range(1,len(p_norm)):
                p_norm[i] += p_norm[i-1]
                pm_norm[i] += pm_norm[i-1]
            return (p_norm + pm_norm*0.5) / 1.5
    
    def getDatas(self,pos):
        pass
    
    def GenerateRoute(self,points,start=0,test=False):
        WifiData = []
        MagData = []
        GPSData = []
        WG_select = []
        labels = []
        label = []
        start = np.random.randint(0,50)
        #w,m,g,WGS = getDatas(start+1,test)
        #WifiData.append(w)
        #MagData.append(m)
        #GPSData.append(g)
        #WG_select.append(WGS)
        labels.append(self.coor[start])
        label.append(start)
        for i in range(points):
            if i ==0:
                momentum = self.coor[start] - self.coor[start]
                NextPos = np.argmax(self.getCDFMap(self.coor[start],1,momentum)>np.random.rand())
                momentum = self.coor[start] - self.coor[NextPos]
            else:
                now = self.coor[NextPos]
                NextPos = np.argmax(self.getCDFMap(self.coor[NextPos],1,momentum)>np.random.rand())
                momentum = now - self.coor[NextPos]
            #w,m,g,WGS = getDatas(NextPos+1,test)
            #WifiData.append(w)
            #MagData.append(m)
            #GPSData.append(g)
            #WG_select.append(WGS)
            labels.append(self.coor[NextPos])
            label.append(NextPos)
            #print(NextPos+1)
        return np.array(label),np.array(labels)

generate_steps = 21
obj = route_generator()
L,L2 = obj.GenerateRoute(generate_steps-1)
print(L.reshape(-1))
print(L.shape)
print(L2)
print(L2.shape)