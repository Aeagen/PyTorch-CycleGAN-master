import tkinter as tk
from tkinter import messagebox
import imageio
from tkinter import filedialog
from PIL import Image, ImageTk

import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch

from models import Generator

class Vis_app():
    def __init__(self,width=700,height=450):
        # 初始化窗口属性
        self.window = tk.Tk()
        self.window.config(bg='#87CEEB')
        self.window.title('CycleGAN循环生成对抗神经网络风格转换--(二次元---真实人像)')
        self.window.geometry(str(width)+'x'+str(height))
        self.img_file = '' # 选择的文件图片
        self.img = '' # 可视化的图片
        self.img_gif0 = tk.PhotoImage(file='')  # 设置原图片
        self.img_gif1 = tk.PhotoImage(file='')  # 设置可视化图片
        self.flag = 0


        # 设置两个显示图片的label
        self.label_img0 = tk.Label(self.window, image=self.img_gif0)  # 设置预显示图片
        self.label_img1 = tk.Label(self.window, image=self.img_gif1)  # 设置可视化图片

        # 解析视频控件
        # lab1 = tk.Label(self.window, text='请输入哔哩哔哩视频OID号：', font=('宋体', 10,), fg='blue', bg='#87CEEB')
        # entry_text = tk.Entry(self.window, textvariable=self.oid)
        model_button_1 = tk.Button(self.window ,text='加载人像转二次元风格', width=25, command=self.load_model_1)
        model_button_2 = tk.Button(self.window, text='加载二次元转人像风格', width=25, command=self.load_model_2)
        btn1 = tk.Button(self.window, text='转换', width=5, command=self.test)


        # 选择图片控件
        lab2 = tk.Label(self.window, text='选择风格转换图片：', font=('宋体', 10,), fg='black', bg='#87CEEB')
        btn2 = tk.Button(self.window, text='选择图片', width=8, command=self.select_img)
        self.lab3 = tk.Label(self.window, text='', bg='#87CEEB')


        lab4 = tk.Label(self.window, text='注：风格转换的图片尺寸推荐(256*256),其他尺寸亦可以\n'
                                          '模型因为训练不完全，效果可能不会太好',fg='red',bg='#87CEEB')

        # 控件布局
        self.label_img0.place(x=50, y=0)
        self.label_img1.place(x=width/2-10, y=0)
        model_button_1.place(x=50, y=300)
        model_button_2.place(x=300, y=300)
        btn1.place(x=550, y=300)
        lab2.place(x=200, y=350)
        btn2.place(x=350, y=350)
        self.lab3.place(x=425, y=350)
        lab4.place(x=200,y=400)

        parser = argparse.ArgumentParser()
        parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
        parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
        parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
        parser.add_argument('--cuda', default=True, help='use GPU computation')
        parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth',
                            help='A2B generator checkpoint file')
        parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth',
                            help='B2A generator checkpoint file')
        self.opt = parser.parse_args()

        self.netG_A2B = Generator(self.opt.input_nc, self.opt.output_nc)
        self.netG_B2A = Generator(self.opt.output_nc, self.opt.input_nc)

        if self.opt.cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()

    def load_model_1(self):
        self.netG_A2B.load_state_dict(torch.load(self.opt.generator_A2B))
        self.netG_A2B.eval()
        self.flag = 0

    def load_model_2(self):
        self.netG_B2A.load_state_dict(torch.load(self.opt.generator_B2A))
        self.netG_B2A.eval()
        self.flag = 1

    def select_img(self):
        # 获取选择对话框中选择的图片
        self.img_file = filedialog.askopenfilename()
        try:
            if self.img_file != '':
                self.img_name = self.img_file.split('/')[-1] # 截取路径最后一段/img.png(如c:/.../img.png
                self.lab3.config(text=self.img_name) # 显示选择的图片名字
                #将用户选择的图片进行尺寸缩放，变为256x256
                image_pil = Image.open(self.img_file).resize((256, 256))
                self.img_gif0 = ImageTk.PhotoImage(image_pil)
                # 显示选择的图片文件
                self.label_img0.config(image=self.img_gif0)
                # 设置传入词语的图片文件
                self.mask = imageio.v2.imread(self.img_file)
            else:
                self.lab3.config(text='暂未选择文件')
                messagebox.showwarning(title='未选择图片', message='请选择一张图片')

        except Exception  as e :
            pass

    def test(self):

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.opt.cuda else torch.Tensor
        input_A = Tensor(1, self.opt.input_nc, self.opt.size, self.opt.size)

        image_pil = Image.open(self.img_file).convert('RGB').resize((256, 256))
        transf = transforms.ToTensor()
        image_pil = transf(image_pil)
        real_A = Variable(input_A.copy_(image_pil))

        if self.flag == 0:
            # Generate output
            fake = 0.5 * (self.netG_A2B(real_A).data + 1.0)
        elif self.flag == 1 :
            fake = 0.5 * (self.netG_B2A(real_A).data + 1.0)
        self.img = './output/shuchu.png'
        save_image(fake, 'output/shuchu.png')
        self.img_gif1 = tk.PhotoImage(file=self.img)
        self.label_img1.config(image=self.img_gif1)


    def main_loop(self):
        self.window.mainloop()


if __name__ == '__main__':
    app = Vis_app()
    app.main_loop()

