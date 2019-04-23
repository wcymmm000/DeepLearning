function [ output_args ] = dice_coef( label_img,seg_img )
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明

seg_img=round(seg_img-0.1);

for i1 = 1:size(label_img,4)
    fenmu=0;
   fenzi=0;
  for i2=1:4
      for i3=1:256
          for i4=1:256
              if  label_img(i2,i3,i4,i1)+seg_img(i2,i3,i4,i1)==2
                  fenzi=fenzi+1;
                  fenmu=fenmu+1;
              end
              if  label_img(i2,i3,i4,i1)+seg_img(i2,i3,i4,i1)==1
                  fenmu=fenmu+1;
              end
          end
      end
  end
  output_args(i1,1)=fenzi/fenmu;
end

end

