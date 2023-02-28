# PerLesionDet_for_Oph-The
# References
main code is from bubbliiiing and thanks him a lot!
https://github.com/bubbliiiing/yolox-pytorch.git
# Usage
1.Prepare ur dataset as VOC2007 format   →    VOCdevkit/VOC2007/JPEGImages(Annotations)   note: Annotations in xml format
2.Prepare ur test/validation dataset → valjpg  valxml
3.Run voc_annotaion.py/voc_annotaiontest.py and 2007_trian.txt/12007_train.txt will be generated for training and validating
4.Run train.py. Some parameters can be changed such as batch_size/input_size/lr...(BTW, I added some attention block in nets/darknet.py and yolo.py, u can choose to shut is down)
5.Get map by get_map.py. Predict by predict.py...
6.If u got any problems, here is my email    leosigmoid@stu.scu.edu.cn  and if u need my wechat for further communication, please identify urself.(I'm pleased for communication and this is just for some information security)


And......If this work may offer any help to your research, please cite our papar~

@article{Wang2023IntelligentDO,
  title={Intelligent Diagnosis of Multiple Peripheral Retinal Lesions in Ultra-widefield Fundus Images Based on Deep Learning},
  author={Tong Wang and Guoliang Liao and Lin Chen and Yan Zhuang and Sibo Zhou and Qiongzhen Yuan and Lin Han and Shanshan Wu and Ke Chen and Binjian Wang and Junyu Mi and Yunxia Gao and Jiangli Lin and Ming Zhang},
  journal={Ophthalmology and Therapy},
  year={2023},
  pages={1 - 15}
}




![微信截图_20230228112341](https://user-images.githubusercontent.com/57985340/221745435-7ea9e403-6009-4977-9bd2-d256a3842eac.png)
