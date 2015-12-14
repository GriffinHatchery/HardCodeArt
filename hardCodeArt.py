from Tkinter import *
from tkFileDialog import *
from PIL import Image, ImageTk
import tkSimpleDialog
from random import *
import os
from os import listdir
from os.path import isfile, join
import datetime
from numpy import *
from numpy.fft import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import cPickle

class ImageViewer:

    def __init__(self, master, filelist,traininglist):
        self.top = master
        self.files = filelist
        self.training = traininglist
        self.fileindex={f: i for i, f in enumerate(filelist)}
        self.filefft={}
        self.save = ""
        self.imsidelen=400
        self.datasidelen=100
        self.datainput=48
        self.windowsize=25
        self.windowarea=self.windowsize*self.windowsize
        self.index = 0
        self.AILook=5
        #self.loadnet()
        self.Emotions = [0] * len(self.files)
        self.menubar = Menu(master)

        menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=menu)
        menu.add_command(label="Save", command=lambda : self.savetraining())
        menu.add_command(label="Save As", command=lambda : self.saveastraining())
        menu.add_command(label="Open", command=lambda :  self.loadtraining())
        menu.add_command(label="New", command=lambda : self.resetselect())
        menu.add_command(label="Add img", command=lambda : self.addimg())
        menu.add_command(label="Train", command=lambda : self.train())
        menu.add_command(label="AI", command=lambda : self.AIread())
        menu.add_command(label="Analyze", command=lambda : self.Analyze())


        master.config(menu=self.menubar)

        filename = filelist[0]

        self.title = Label(text=os.path.basename(filename))
        self.title.pack()

        # the image frame
        imfr = Frame(master)
        imfr.pack(side='top', expand=1, fill='x')

        im = self.format_image(Image.open(filename))
        self.tkimage = ImageTk.PhotoImage(im)

        self.lbl = Label(imfr, image=self.tkimage)
        self.lbl.pack(side='left')
        self.tkfftimage = ImageTk.PhotoImage(self.fftimage(im))

        self.fftlbl = Label(imfr, image= self.tkfftimage);
        self.fftlbl.pack(side='right')

        # the navigation frame
        frnav = Frame(master)
        frnav.pack(side='top', expand=1, fill='x')

        backbn = Button(frnav, text="back", command=lambda : self.toframe(self.index-1))
        backbn.grid(row=0, column=0, sticky="w", padx=4, pady=4)

        ilabel = Label(frnav, text="image number:")
        ilabel.grid(row=0, column=1, sticky="e", pady=4)

        self.evar = IntVar()
        self.evar.set(1)
        entry = Entry(frnav, textvariable=self.evar)
        entry.grid(row=0, column=2, sticky="w", pady=4)
        entry.bind('<Return>', self.getimgnum)

        nextbn = Button(frnav, text="next", command=lambda : self.toframe(self.index+1))
        nextbn.grid(row=0, column=3, sticky="e", padx=4, pady=4)

        findbn = Button(frnav, text="find", command=lambda : self.toframe(self.find()))
        findbn.grid(row=0, column=4, sticky="e", padx=4, pady=4)

        """
        # the file frame
        frfile = Frame(master)
        frfile.pack(side='top', expand=1, fill='x')

        savebn = Button(frfile, text="save", command=lambda : self.savetraining())
        savebn.grid(row=0, column=0, sticky="e", padx=4, pady=4)

        saveasbn = Button(frfile, text="save as", command=lambda : self.saveastraining())
        saveasbn.grid(row=0, column=1, sticky="e", padx=4, pady=4)

        loadbn = Button(frfile, text="load", command=lambda : self.loadtraining())
        loadbn.grid(row=0, column=2, sticky="e", padx=4, pady=4)

        resetbn = Button(frfile, text="start over", command=lambda : self.resetselect())
        resetbn.grid(row=0, column=4, sticky="e", padx=4, pady=4)

        """
        # CheckBox frame
        cbfr = Frame(master)
        cbfr.pack(side='top', expand=1, fill='x')

        self.cbs=[]
        self.cbvars=[]
        self.cbtext=["Sad","Happy","Serious","Whimsical","Peaceful","Intense","Angry","Simple"]
        i = 0
        j = 0
        for str in self.cbtext:
            var=IntVar()
            self.cbvars.append(var)
            cb = Checkbutton(cbfr, text=str, variable=var)
            cb.grid(row=j, column=i, sticky="w", padx=4, pady=4)
            self.cbs.append(cb)
            i+=1
            if i > 3:
                i = 0
                j += 1

    def find(self):
        for i in range(self.index+1, len(self.Emotions)):
            if self.Emotions[i] is 0:
                return i

        for i in range(0,self.index):
            if self.Emotions[i] is 0:
                return i

        return self.index

    def toframe(self,imgnum):
        self.getcbselect()
        self.index = imgnum
        if self.index >= len(self.files):
            self.index = 0
        elif self.index < 0:
            self.index = len(self.files) - 1

        filename = self.files[self.index]

        self.title.configure(text=os.path.basename(filename))

        self.evar.set(self.index+1)

        im = self.format_image(Image.open(filename))
        self.tkimage.paste(im)
        self.tkfftimage.paste(self.fftimage(im))
        self.setselect()

    def setselect(self):
        num=self.Emotions[self.index]
        for i in range(0, len(self.cbs)):
            self.cbvars[i].set(num >> i & 1)

    def getcbselect(self):
        num=0
        for i in range(0, len(self.cbvars)):
            num+=self.cbvars[i].get() << i
        self.Emotions[self.index]=num

    def resetselect(self):
        for i in range(0, len(self.Emotions)):
            self.Emotions[i]=0
        self.setselect()
        self.toframe(0)
        self.save = ""

    def getimgnum(self, event=None):
        self.toframe(self.evar.get()-1)

    def format_image(self,im):
        width, height = im.size
        m=min(height, width)
        left = (width-m)/2
        top = (height-m)/4
        right = m+ left
        bottom = m+ top
        return im.crop((left, top, right, bottom)).resize((self.imsidelen,self.imsidelen))

    def savetraining(self):
        self.getcbselect()
        if self.save is "":
            s=self.getname()
            if s is None:
                return
            self.save=s
        out = open(self.save,'w')
        for i in range(0, len(self.files)):
          if self.Emotions[i] is not 0:
              out.write(self.files[i] + " " +str(self.Emotions[i]) + "\n" )
        out.close()

    def saveastraining(self):
        s=self.getname(0)
        if s is None or s is "":
            return
        self.save=s
        self.savetraining()

    def loadtraining(self):
        s=self.getname(0)
        if s is None or s is "":
            return
        self.Emotions = [0] * len(self.files)
        self.save=s
        inf = open(s,'r')
        for line in inf:
            val = line.split(' ', 2)
            self.Emotions[self.fileindex[val[0]]]=int(val[1])

        inf.close()
        self.setselect()


    def getname(self,auto=1):
        nd=NameDialog(self.top)
        name = nd.result
        if name is None:
            return None
        if auto is not 1:
            return 'trainings/'+name+'.txt'

        if name is "":
            name = datetime.date.today().strftime("%m_%d_%y")

        if not isfile('trainings/'+name+'.txt'):
            return 'trainings/'+name+'.txt'
        i = 1
        while isfile('trainings/'+name+'_'+str(i)+'.txt'):
            i += 1
        return 'trainings/'+name+'_'+str(i)+'.txt'


    def loadnet(self):
        if(isfile("net.p")):
            self.net = cPickle.load( open( "net.p", "rb" ) )
        else:
            self.newnet()

    def newnet(self):
        self.net = buildNetwork(self.datainput, 16, 8, bias=True)

    def savenet(self):
        cPickle.dump(self.net, open( "net.p", "wb" ))

    def img_toarray(self, im):
        imdata = im.load()
        width, height = im.size
        return [[[imdata[i,j][k] for i in range(width)] for j in range(height)]for k in range(3)]


    def img_fft(self, im):
        ary=self.img_toarray(im)
        return [abs(fft2(ary[k])) for k in range(3)]

    def fftimagetest(self,im):
        i= im.convert('L')    #convert to grayscale
        a = asarray(i)

        b = (fft2(a))*1j
        b = abs(ifft2(b))
        return Image.fromarray(b.astype(uint8))

    def fftimage(self,im):
        imfft = self.img_fft(im)
        width, height = im.size
        advr=1.0/(width*height*3)
        adv=0.0
        advsq=0.0
        for j in range(width):
            for i in range(height):
                for k in range(3):
                    s=imfft[k][i][j]*advr
                    adv+=s
                    advsq+= imfft[k][i][j]*s

        std=sqrt(advsq-adv*adv)

        delta=.5*std
        low=adv-delta
        ratio= 255/(2*delta)

        for j in range(width):
            for i in range(height):
                for k in range(3):
                    imfft[k][i][j]=(imfft[k][i][j]-low)*ratio
                    if imfft[k][i][j] > 255:
                        imfft[k][i][j] = 255
                    if imfft[k][i][j] < 0:
                        imfft[k][i][j] = 0


        data = zeros((width,height,3), dtype=uint8)
        for j in range(width):
            for i in range(height):
                for k in range(3):
                    data[i, j][k]=imfft[k][i][j]
        img = Image.fromarray(data, 'RGB')
        return img

    def fftfile(self, filename):
        if(not self.filefft.has_key(filename)):
            im = self.format_image(Image.open(filename))
            s = self.datasidelen*choice((1,2,4))
            #s = self.datasidelen
            imsmall=im.resize((s,s))
            i = randrange(s-self.datasidelen+1)
            j = randrange(s-self.datasidelen+1)
            imsmall=imsmall.crop((i,j,i+self.datasidelen,j+self.datasidelen))
            self.filefft[filename]= self.keyData(self.img_fft(imsmall))
        return self.filefft[filename];


    def keyData(self,fft):
        ret=[0.0]*self.datainput
        index =0
        for ii in range(2):
            for jj in range(4):
                for k in range(3):
                    for j in range(self.windowsize):
                        for i in range(self.windowsize):
                            d=fft[k][i+ii*self.windowsize][j+jj*self.windowsize];
                            #average
                            ret[index]+=d/(self.windowarea)
                            #average square
                            ret[index+1]+=d*d/(self.windowarea)

                    index += 2


        index =0
        for ii in range(2):
            for jj in range(4):
                for k in range(3):
                    #standard deviations
                    ret[index+1]=sqrt(abs(ret[index+1]-ret[index]*ret[index]))
                    index += 2


        return ret

    def tobit(self, num):
        bit=[0]*8
        for i in range(8):
            bit[i]=(num >> i & 1)
        return bit

    def tonum(self,bit):
        num=0
        j=-1
        m=0
        for i in range(0, len(bit)):
            if bit[i] >= .5:
                num+=1 << i
            if bit[i] > m:
                m=bit[i]
                j=i
        if num <= 0:
            num=1 << j
        return num

    def addimg(self):
        filename = askopenfilename()
        self.files.append(filename)
        self.Emotions.append(0)
        self.toframe(-1)

    def gettraining(self):
        DS = SupervisedDataSet(self.datainput, 8)
        for trn in self.training:
            inf = open(trn,'r')
            for line in inf:
                val = line.split(' ', 2)
                index = self.fileindex[val[0]]
                if index>=10:
                    input=self.fftfile(val[0])
                    output=self.tobit(int(val[1]))
                    DS.appendLinked(input, output)
            inf.close()
        return DS

    def train(self):
        self.loadnet()
        while True:
            self.filefft={}
            ds=self.gettraining()
            trainer = BackpropTrainer(self.net,ds)
            print trainer.train()
            self.savenet()

    def AIread(self):
        bit = self.AIgetBits(self.index)
        print bit
        self.Emotions[self.index]=self.tonum(bit)
        self.setselect()

    def AIgetBits(self,index):
        self.loadnet()
        file=self.files[index]
        bit=[0]*8

        for i in range(self.AILook):
            fft=self.fftfile(file)
            bit=add(bit,self.net.activate(fft))
        return divide(bit,self.AILook)

    def Analyze(self):
        sd=[0.0] * len(self.files)
        mean=[0.0] * len(self.files)
        num=[0] * len(self.files)

        aisd=[0.0] * len(self.files)
        aimean=[0.0] * len(self.files)
        ainum=[0] * len(self.files)

        emotions = [[0 for i in range(len(self.files))] for i in range(len(self.training))]
        aiemotions=[0] * len(self.files)
        for tri in range(len(self.training)):
            trn=self.training[tri]
            inf = open(trn,'r')
            for line in inf:
                val = line.split(' ', 2)
                index = self.fileindex[val[0]]
                emotions[tri][index]=int(val[1])
            inf.close()

        for fi in range(len(self.files)):
            eai= self.tonum(self.AIgetBits(fi));
            aiemotions[fi]=eai
            for tri in range(len(self.training)):
                ei=emotions[tri][fi]
                if(ei == 0):
                    continue

                for trj in range(tri):
                    ej=emotions[trj][fi]
                    if(ej == 0):
                        continue
                        
                    num[fi]+=1
                    c = self.count(ei ^ ej)
                    mean[fi]+=c
                    sd[fi]+=c*c


                ainum[fi]+=1
                aic = self.count(ei ^ eai)
                aimean[fi]+=aic
                aisd[fi]+=aic*aic

        for fi in range(len(self.files)):
            mean[fi]/=num[fi]
            sd[fi]=sqrt(abs(sd[fi]/num[fi]-mean[fi]*mean[fi]))

            aimean[fi]/=ainum[fi]
            aisd[fi]=sqrt(abs(aisd[fi]/ainum[fi]-aimean[fi]*aimean[fi]))

        out = open("analyze.txt",'w')

        out.write("file\tmean\tsd\tAi's Mean\tAi's sd\n")
        for fi in range(len(self.files)):
            out.write(self.files[fi] + "\t" + str(mean[fi]) + "\t" + str(sd[fi]) + "\t" + str(aimean[fi]) + "\t" + str(aisd[fi]) + "\n")
        out.close()


    def count(self,num):
        ans = 0
        for i in range(8):
            ans += (num >> i & 1)
        return ans


class NameDialog(tkSimpleDialog.Dialog):

    def body(self, master):
        self.result=None
        Label(master, text="Enter a name or initials:").grid(row=0)
        self.n = Entry(master)
        self.n.grid(row=1)
        return self.n

    def apply(self):
        self.result = str(self.n.get())


if __name__ == "__main__":

    filelist = [ join('images', f) for f in listdir('images') if isfile(join('images',f)) ]
    traininglist = [ join('trainings', f) for f in listdir('trainings') if isfile(join('trainings',f)) ]

    root = Tk()
    app = ImageViewer(root, filelist,traininglist)
    root.mainloop()


