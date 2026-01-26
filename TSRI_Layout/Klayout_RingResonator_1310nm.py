import gdsfactory as gf
from gdsfactory.gpdk import PDK
import numpy as np

gf.gpdk.PDK.activate()

#-------------參數-------------
wavelength=1310 # operation wavelength
TopLength=3.905
TopRadius=5.770
BotLength=3.641
BotRadius=7.835
RingGap=0.28
waveguide_width=0.38
gap=0.2

#對應的heater寬度(nm)
TopRingHeaterWidth=2000

npointss=20000

#reverse waveguide parameters
Port1width=0.55
Port2width=8
Port3width=12.7
Port1hight=4
Port3hight=11


#-------------參數-------------

#------------簡化顯示的數字們------------

BotWaveguide_ycenter=-(waveguide_width*2+gap+TopRadius)-(TopRadius+BotRadius+RingGap+2*waveguide_width)*np.cos(60*np.pi/180)-BotRadius-waveguide_width-gap-waveguide_width/2
TotalLength=TopLength+BotLength+2*TopRadius+2*BotRadius+RingGap*np.cos(30*np.pi/180)+waveguide_width*4

#圓心連線長度
O2toO3=TopRadius+RingGap+BotRadius+waveguide_width

L1=[ TopRadius , -waveguide_width-gap ]
L2=[ TopRadius , -waveguide_width-gap-TopRadius*2 ]
L3=[ TopRadius+TopLength+O2toO3*np.sin(60*np.pi/180.) , -waveguide_width-gap-TopRadius-(O2toO3)*np.cos(60*np.pi/180.)+BotRadius ]
L4=[ TopRadius+TopLength+O2toO3*np.sin(60*np.pi/180.) , -waveguide_width-gap-TopRadius-(O2toO3)*np.cos(60*np.pi/180.)-BotRadius ]


#調整大小環Heater的覆蓋比例
BotHeaterLengthOg=BotLength+np.pi*BotRadius
TopHeaterLengthOg=TopLength+np.pi*TopRadius
HeaterCoverageRatio=(BotHeaterLengthOg)/(BotLength*2+2*np.pi*BotRadius)-(TopHeaterLengthOg)/(TopLength*2+2*np.pi*TopRadius)
BotHeaterChange=HeaterCoverageRatio*(BotLength*2+2*np.pi*BotRadius)
print("測試覆蓋比例是否相等：",(BotHeaterLengthOg-BotHeaterChange)/(BotLength*2+2*np.pi*BotRadius)-(TopHeaterLengthOg)/(TopLength*2+2*np.pi*TopRadius))

BotHeaterLength=BotHeaterLengthOg-BotHeaterChange
TopHeaterLength=TopHeaterLengthOg

BotRingHeaterWidth=TopRingHeaterWidth*BotHeaterLength/TopHeaterLength
print("BotRingHeaterWidth=",BotRingHeaterWidth)

#------------簡化顯示的數字們------------

c = gf.Component(f"Ring_{wavelength}")

#-------------產生圖形的function-------------
def waveguide(Components,Length,x,y,a,b):#產生直線波導，a,b決定layer
    p1 = Components.add_polygon([(x, y), (x+Length, y), (x+Length, y-waveguide_width), (x, y-waveguide_width)], layer=(a, b))
    return p1


def circle1(Radius,a,b):
    c = gf.components.bend_circular(radius=Radius, width=waveguide_width,allow_min_radius_violation=True, angle=180, npoints=npoints, layer=(a, b))
    return c

def Port(a,b):#產生反向波導，a,b決定layer
    R = gf.components.rectangle(size=(Port3width, Port3hight+waveguide_width), layer=(1, 0))
    T = gf.components.taper(length=Port2width, width1=Port1hight+waveguide_width, width2=Port3hight+waveguide_width, layer=(2, 0))
    A = gf.components.rectangle(size=(Port1width, Port1hight+waveguide_width), layer=(4, 0))

    T_shifted = gf.Component()
    t_ref = T_shifted.add_ref(T)
    t_ref.move((-Port2width,((Port3hight+waveguide_width))/2))
    C = gf.boolean(A=R, B=T_shifted, operation="or", layer1=(1, 0), layer2=(2, 0), layer=(3, 0))

    A_shifted = gf.Component()
    a_ref = A_shifted.add_ref(A)
    a_ref.move((-Port1width-Port2width,(Port3hight-Port1hight)/2))
    D = gf.boolean(A=C, B=A_shifted, operation="or", layer1=(3, 0), layer2=(4, 0), layer=(a, b))
    return D

def electrode(x1,y1):#產生電極，x1,y1為電極接觸WG位置中心點，x2,y2為電極中心點，n1,n2決定連接時使用的點(左上開始順時針1234)
    p1 = c.add_polygon([(x1-1, y1+1), (x1+1, y1+1), (x1+1, y1-1), (x1-1, y1-1)], layer=(120, 0))#寬度為1的正方體
    p2 = c.add_polygon([(x1-2.5, y1+2.5), (x1+2.5, y1+2.5), (x1+2.5, y1-2.5), (x1-2.5, y1-2.5)], layer=(125, 0))#寬度為5的正方體
    p3 = c.add_polygon([(x1-2.5, y1+2.5), (x1+2.5, y1+2.5), (x1+2.5, y1-2.5), (x1-2.5, y1-2.5)], layer=(115, 0))#寬度為5的正方體
    return p1, p2, p3



def HeaterRegionOG(Components, Width , i ):#產生加熱區域，Width為加熱區域寬度，x1,x2為加熱區域長度範圍，y為加熱區域y位置
    t1 = Components.get_region(layer=(10, 0))
    t2 = t1.sized(Width/2-waveguide_width*1000/2)
    RorL=[[(L1[0]+TopLength, 10000), (L1[0]+TopLength+10000, 10000), (L1[0]+TopLength+10000, -10000), (L1[0]+TopLength, -10000)],[(L3[0]-10000, 10000), (L3[0], 10000), (L3[0], -10000), (L3[0]-10000, -10000),],
          [(L1[0],L1[1]+TopRingHeaterWidth/2000+1),(L1[0]+TopLength,L1[1]+TopRingHeaterWidth/2000+1),(L1[0]+TopLength,L1[1]-TopRingHeaterWidth/2000-1),(L1[0],L1[1]-TopRingHeaterWidth/2000-1)],[(L4[0],L4[1]+10000),(L4[0]+BotLength,L4[1]+10000),(L4[0]+BotLength,L4[1]-BotRingHeaterWidth/2000-1),(L4[0],L4[1]-BotRingHeaterWidth/2000-1)]]
    t3_boolen = gf.Component()
    t3_boolen.add_polygon(RorL[i], layer=(1, 0))
    t2_boolen = gf.Component()
    t2_boolen.add_polygon(t2, layer=(2, 0))
    t4 = gf.boolean(A=t2_boolen, B=t3_boolen, operation="not", layer1=(2, 0), layer2=(1, 0), layer=(115, 0))
    t5_boolen = gf.Component()
    t5_boolen.add_polygon(RorL[i+2], layer=(5, 0))
    t6 = gf.boolean(A=t4, B=t5_boolen, operation="not", layer1=(115, 0), layer2=(5, 0), layer=(115, 0))
    return t6

#-------------產生圖形的function-------------


#-------------產生圖形---------------------

#TOP Bus waveguide
waveguide(c,TotalLength+90,-45,0,10,0)

npoints=5000
#TOP Ring Resonator Heater region
TopRingResonatorComponents = gf.Component()
Top1 = waveguide(TopRingResonatorComponents,TopLength,L1[0],L1[1],10,0)
Top2 = TopRingResonatorComponents.add_ref(circle1(TopRadius,10,0)).mirror_x().move((L2[0],L2[1]-waveguide_width/2))
Top3 = TopRingResonatorComponents.add_ref(circle1(TopRadius,10,0)).move((L2[0]+TopLength,L2[1]-waveguide_width/2))
Top4 = waveguide(TopRingResonatorComponents,TopLength,L2[0],L2[1],10,0)
c << HeaterRegionOG(TopRingResonatorComponents, TopRingHeaterWidth , 0)

npoints=npointss
TopRingResonatorComponents1 = gf.Component()
Top1 = waveguide(TopRingResonatorComponents1,TopLength,L1[0],L1[1],10,0)
Top2 = TopRingResonatorComponents1.add_ref(circle1(TopRadius,10,0)).mirror_x().move((L2[0],L2[1]-waveguide_width/2))
Top3 = TopRingResonatorComponents1.add_ref(circle1(TopRadius,10,0)).move((L2[0]+TopLength,L2[1]-waveguide_width/2))
Top4 = waveguide(TopRingResonatorComponents1,TopLength,L2[0],L2[1],10,0)
c << TopRingResonatorComponents1


npoints=5000
#BOT Ring Resonator Heater region
BotRingResonatorComponents = gf.Component()
Bot1 = waveguide(BotRingResonatorComponents,BotLength,L3[0], L3[1],10,0)
Bot2 = BotRingResonatorComponents.add_ref(circle1(BotRadius,10,0)).mirror_x().move((L4[0],L4[1]-waveguide_width/2))
Bot3 = BotRingResonatorComponents.add_ref(circle1(BotRadius,10,0)).move((L4[0]+BotLength,L4[1]-waveguide_width/2))
Bot4 = waveguide(BotRingResonatorComponents,BotLength,L4[0],L4[1],10,0)
c << HeaterRegionOG(BotRingResonatorComponents, BotRingHeaterWidth , 1)

npoints=npointss
BotRingResonatorComponents1 = gf.Component()
Bot1 = waveguide(BotRingResonatorComponents1,BotLength,L3[0], L3[1],10,0)
Bot2 = BotRingResonatorComponents1.add_ref(circle1(BotRadius,10,0)).mirror_x().move((L4[0],L4[1]-waveguide_width/2))
Bot3 = BotRingResonatorComponents1.add_ref(circle1(BotRadius,10,0)).move((L4[0]+BotLength,L4[1]-waveguide_width/2))
Bot4 = waveguide(BotRingResonatorComponents1,BotLength,L4[0],L4[1],10,0)
c << BotRingResonatorComponents1

#BOT Bus waveguide
waveguide(c,TotalLength+90,-45,L4[1]-gap-waveguide_width,10,0)

c.add_polygon([(L3[0]+BotHeaterChange,L3[1]+BotRingHeaterWidth/2000-waveguide_width/2),(L3[0]+BotLength,L3[1]+BotRingHeaterWidth/2000-waveguide_width/2),(L3[0]+BotLength,L3[1]-BotRingHeaterWidth/2000-waveguide_width/2 ),(L3[0]+BotHeaterChange,L3[1]-BotRingHeaterWidth/2000-waveguide_width/2)], layer=(115, 0))


#電極
electrode(L1[0]-2.5,L1[1]-TopRingHeaterWidth/2000-waveguide_width/2+2.5)
electrode(L2[0]+TopLength-2.5,L2[1]+TopRingHeaterWidth/2000-waveguide_width/2-2.5)
electrode(L3[0]+2.5,L3[1]-BotRingHeaterWidth/2000-waveguide_width/2+2.5)
electrode(L4[0]+BotLength+2.5,L4[1]+BotRingHeaterWidth/2000-waveguide_width/2-2.5)

#避免銳角
c.add_polygon([(L1[0]-5,L1[1]-TopRingHeaterWidth/2000-waveguide_width/2),(L1[0]-4,L1[1]-TopRingHeaterWidth/2000-waveguide_width/2),(L1[0]-4,L1[1]-TopRingHeaterWidth/2000-waveguide_width/2-1),(L1[0]-5,L1[1]-TopRingHeaterWidth/2000-waveguide_width/2-1)], layer=(115, 0))

#-------------產生圖形---------------------


c.plot()
c.write_gds(f"Ring_{wavelength}.gds")  # Write it to a GDS file. You can open it in klayout.
c.show()  # Show it in klayout.
  # Plot it in jupyter notebook.