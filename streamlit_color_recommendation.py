import streamlit as st
import numpy as np
import cv2 as cv
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from PIL import ImageColor
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd
import colorsys

@st.cache
def rgb_to_hex(rgb):
    #change rgb values to hex values
    return '%02x%02x%02x' % rgb

@st.cache
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

@st.cache
def color_rec(colors):
  i = 0
  new = np.array([])
  RGB = np.array([])

  for each in colors:
    temp = hex_to_rgb(each)
    #print(temp)
    coor = colorsys.rgb_to_hsv(temp[0],temp[1],temp[2])
    #print(coor)
    if i==0:
      new = coor
      RGB = temp
    else:
      #print(RGB)
      new = np.vstack((new,coor))
      RGB = np.vstack((RGB,temp))
    i=i+1
    #pick top two colors. Offer recommendations based on analogous triadic and tetradic
  angle1 = new[0][0]
  angle2 = new[1][0]
  suggest = [[.0, .0, .0], 
           [.0, .0, .0]] 
  #first is the triadic recommendation
  #second is analogous

  #print(angle1)

  #print(angle2)

  dif = abs(angle1-angle2)

  #print(dif)
  suggest[0][1] = (new[0][1]+new[1][1])/2
  suggest[1][1] = (new[0][1]+new[1][1])/2
  suggest[0][2] = (new[0][2]+new[1][2])/2
  suggest[1][2] = (new[0][2]+new[1][2])/2

  #print(suggest)

  if dif > 0.5:
    #print("big")
    this = abs(((new[0][0]+new[1][0])/2))
    that = abs(((new[0][0]+new[1][0])/2)+0.5)
    suggest[0][0] = this
    suggest[1][0] = that
  elif dif < 0.5:
    #print("smol")
    this = abs(((new[0][0]+new[1][0])/2))
    that = abs(((new[0][0]+new[1][0])/2)+0.5)
    if that >= 1:
      that = that-1
    suggest[0][0] = that
    suggest[1][0] = this
  else:
    print("there is something awfuly wrong")
  #print(suggest)
  R,G,B = colorsys.hsv_to_rgb(suggest[0][0],suggest[0][1],suggest[0][2])
  other = [(np.int(np.round(R))), (np.int(np.round(G))), (np.int(np.round(B)))]
  fRGB = other
  #print(RGB)
  #print(fRGB)
  R,G,B = colorsys.hsv_to_rgb(suggest[1][0],suggest[1][1],suggest[1][2])
  other = [(np.int(np.round(R))), (np.int(np.round(G))), (np.int(np.round(B)))]
  fRGB = np.vstack((fRGB,other))
  
  #print(fRGB)

  hex_recs = rgb_to_hex((fRGB[0][0],fRGB[0][1],fRGB[0][2]))
  iHopeChrisDoesntReadThroughThisGodForsakenCode = rgb_to_hex((fRGB[1][0],fRGB[1][1],fRGB[1][2]))
  hex_recs = np.stack((hex_recs,iHopeChrisDoesntReadThroughThisGodForsakenCode))

  return hex_recs      
    
@st.cache
def color_id(image, clusters): 
    r = []
    g = []
    b = []
    #add all r,g,b data to array
    for row in image:
        for temp_r, temp_g, temp_b,temp in row:
            if temp!=0 or (temp_r!=0 or temp_g!=0 or temp_b!=0):
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)
    
    #make into dataframe
    image_df = pd.DataFrame({'red' : r,
                              'green' : g,
                              'blue' : b})
    
    #standardize
    image_df['scaled_color_red'] = whiten(image_df['red'])
    image_df['scaled_color_blue'] = whiten(image_df['blue'])
    image_df['scaled_color_green'] = whiten(image_df['green'])
    
    #find clusters
    cluster_centers, _ = kmeans(image_df[['scaled_color_red',
                                        'scaled_color_blue',
                                        'scaled_color_green']], clusters)
     
    dominant_colors = []
    
    #find std dev
    red_std, green_std, blue_std = image_df[['red',
                                              'green',
                                              'blue']].std()
    
    #output dominant colors
    for cluster_center in cluster_centers:
        red_scaled, green_scaled, blue_scaled = cluster_center
        dominant_colors.append((
            int(red_scaled * red_std),
            int(green_scaled * green_std),
            int(blue_scaled * blue_std)
            
        ))
        
    return dominant_colors
    
def color_change(image,original,updated_color):   
    #change to RGBA
    picture=cv.cvtColor(original, cv.COLOR_RGB2RGBA)

    #make new color into correct format
    updated_color=np.asarray(updated_color)
    updated_color=updated_color.tolist()

    # Get the size of the image
    dimensions = picture.shape

    #process every pixel, add new color and make background original


#process every pixel, add new color and make background original
    for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                current_color=image[i,j]
                if current_color[0]==0 and current_color[1]==0 and current_color[2]==0 and current_color[3]==255:
                    new_color=original[i,j]
                    image[i,j]=(0,0,0,0)
                else:
                    new_color=updated_color
                    new_color[3]=255
                picture[i,j]=(new_color)

    gray = cv.cvtColor(image,cv.COLOR_RGBA2GRAY)
    gray = cv.merge((gray.copy(), gray.copy(), gray.copy(), gray.copy()))

    #blend new with original for lighting
    final=cv.addWeighted(picture,0.75,gray,0.25,50)


    

    ################
    # def new_color_change(image, original, updated_color):
    #     new_image = image.copy()
    #     new_image[np.where((image != [0, 0, 0]).all(axis=2))] = updated_color
    #     result = cv2.addWeighted(new_image, 0.5, image, 0.5, 0)
    #
    #     gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #
    #     arr = np.zeros((np.shape(original)))
    #     print(np.shape(arr))
    #     for row in range(np.shape(gray)[0]):
    #         for col in range(np.shape(gray)[1]):
    #             if gray[row][col] != 0:
    #                 original[row][col][:] = result[row][col][:]
    #     cv2.imshow('im', original)

    return final


    
    #gray = cv.cvtColor(image,cv.COLOR_RGBA2GRAY)
    #gray = cv.merge((gray.copy(), gray.copy(), gray.copy(), gray.copy()))

    #st.write(np.shape(gray))
    #blend new with original for lighting

   
@st.cache
def mask_rect(image,rect):
    #Grabcut
    #original=cv.cvtColor(original, cv.COLOR_BGR2RGBA)

    #remove background with grabcut
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    #grabcut
    cv.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = image*mask2[:,:,np.newaxis]

    return image

def mask_free(image, rect):
    #Object before using simple Rectangular Grabcut
    mask = np.zeros(image.shape[:2], np.uint8)  # initialize the layers for the rectangular mask
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    # establishes the mask using grabcut with a rectangle
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image1 = image * mask2[:, :, np.newaxis]  # saves a separate image for the user to view the simple grabcut
    cv.imwrite("image1.png", image1)

    # Using Free Drawings to create a Grabcut Mask with Streamlit
    # Create a canvas component
    # Allow users to draw on the original image to ensure all aspects are maintained

    st.write(
        "Draw red on the objects you want REMOVED from the object pictured below. Click 'Proceed' when you have finished drawing.")
    canvas_result2 = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color= "#FF0000",#st.sidebar.selectbox("Color:", ("#FF0000", "#FFFFFF")),
        # Need to determine how to label these (Delete and Keep)
        background_color="#eeeeee",
        background_image=Image.open("image1.png"),
        update_streamlit=True,
        height=h, width=w,
        drawing_mode="freedraw",
        key="canvas2",
    )
    submit = st.checkbox('Proceed')  # creates a streamlit button for users to proceed with the advanced grabcut

    if submit:# (begin_free_grabcut == 'Yes'):
        newmask = canvas_result2.image_data  # Uses the drawing from the canvas for the new mask layer for the Grabcut
        #st.image(newmask, width=int(w))  # Displays the Drawn Data
        cv.imwrite("new_image_mask_grabcut.png", newmask)  # saves the newmask as a PNG
        newmask = cv.imread("new_image_mask_grabcut.png", 0)  # reads the new mask as a binary image
        newmask = cv.resize(newmask, (image.shape[1], image.shape[0]))  # Resizes the user drawn mask to match
        # the original scale of the image, in order to avoid losing image data

        #Overlay the User drawings on a gray background
        gray_background = Image.new(mode="RGB", size=(image.shape[1], image.shape[0]), color=90)
        gray_background.save("gray_background.png")
        gray_background = cv.imread("gray_background.png", 0)  # Problem Child
        newmask = cv.addWeighted(newmask, 1, gray_background, 1, 0)  # Combines the gray background with the user drawn mask
        cv.imwrite("new_image_mask_grabcut.png", newmask)

        # Create the mask to be used for the grabcut
        # Where marked red (ensure background), change mask = 0
        # mask[newmask == 255] = 1  # Where marked white (ensure foreground), change mask = 1
        # However, this code is currently set to only use the ensure background
        for i in range(np.shape(newmask)[0]):
            for j in range(np.shape(newmask)[1]):
                if newmask[i][j] == 255:
                    mask[i][j]  = 1
                elif newmask[i][j] != 90 & newmask[i][j] != 255:
                    mask[i][j] = 0

        mask, bgdModel, fgdModel = cv.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask[:, :, np.newaxis]

        #insert an st.stop() in an else statement so that the code does not continue until the user has verified they are done with their code

    return image

@st.cache
def background_only(image,original):
    #convert to RGBA
    picture=cv.cvtColor(image, cv.COLOR_BGR2RGBA)

    # Get the size of the image
    dimensions = picture.shape
    #define new color
    new_color=np.zeros(4)
    
    # Process every pixel, change black to transparent white,
    # adjust color of everything else
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            current_color=picture[i,j]
            if current_color[0]==0 and current_color[1]==0 and current_color[2]==0 and current_color[3]==255:
                new_color=original[i,j]
            else:
                new_color=(0,0,0,0)
            picture[i,j]=(new_color)

    return picture

#main
st.title('Furniture Color Change App')

uploaded_img=st.file_uploader("Choose an image...", type=['png','jpg','jpeg'])  # JPEG dont work yet :(


if uploaded_img is not None:
    #change file into decoded image
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)

    #change original into RGBA
    original=cv.cvtColor(image, cv.COLOR_BGR2RGBA)
    cv.imwrite("original.png",original)
    cv.imwrite("picture.png",image)
    
    st.write("Choose the object that you would like to change")
    st.write("Draw 1 rectangle around the object")
    
    #get dimensions and ratio of size, define size for canvas
    dimensions=image.shape
    ratio=dimensions[1]/dimensions[0]
    if ratio>1:
        w=600
        h=600/ratio
    else:
        h=400
        w=400/ratio
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#000000",
        background_color="#eeeeee",
        background_image=Image.open("picture.png"),
        update_streamlit=True,
        height=h, width=w,
        drawing_mode="rect",
        key="canvas",
    )
    
    #define scale
    scalex=dimensions[1]/w
    scaley=dimensions[0]/h
    
    check=st.checkbox("Push this button to continue")
    if check:
        if canvas_result.json_data["objects"]==[]:
            st.write("Please choose an object in the image above")
            check2=False
        #define rectangle object dimensions
        rect_obj=canvas_result.json_data["objects"][-1]
        left=rect_obj["left"]
        top=rect_obj["top"]
        width=rect_obj["width"]
        height=rect_obj["height"]
        rect = (int(left * scalex), int(top * scaley), int(width * scalex), int(height * scaley))
        grabcut_selection = st.radio("What method of object re-coloring would you like to proceed with?",
                                     ('Simple Object Coloring', 'Advanced Custom Coloring'))
        if grabcut_selection == 'Simple Object Coloring':
            rectangle_mask= mask_rect(image, rect)
            st.image(rectangle_mask, channels='BGR', width=int(w))
            mask_1 = rectangle_mask
            canvas_result.json_data["objects"] = []
        # Custom Grabcut with interactive foreground extraction
        elif grabcut_selection == 'Advanced Custom Coloring':
            # Begin Grabcut Free
            free_draw_mask = mask_free(image, rect)
            # Show the image after using grabcut free
            st.write("Object After Advanced Object Extraction")
            st.image(free_draw_mask,channels='BGR', width=int(w))
            mask_1 = free_draw_mask
            canvas_result.json_data["objects"] = []

        #Pick your own color
        check1=st.radio("What would you like to do next?",('Pick my own color','Color Recommendation'))
        if check1=='Pick my own color':
            color=st.color_picker('Pick A Color','#00f900')
            new_color=ImageColor.getcolor(color, "RGBA")
        
        #color recommendation
        if check1=='Color Recommendation':
            #st.write("Loading...")

            #perform grabcut
            #mask_1=mask_rect(image,int(left*scalex),int(top*scaley),int(width*scalex),int(height*scaley))
            
            #make into background only
            background=background_only(mask_1,original)
            st.image(background) #REMOVE
            #output dominant colors in columns
            st.write('Here are the dominant colors from the rest of the image:')
            dom_color=color_id(background,5)
            dom_1=str(rgb_to_hex(dom_color[0]))
            dom_2=str(rgb_to_hex(dom_color[1]))
            dom_3=str(rgb_to_hex(dom_color[2]))
            dom_4=str(rgb_to_hex(dom_color[3]))
            dom_5=str(rgb_to_hex(dom_color[4]))
            st.write(dom_1)
            st.write(dom_2)
            st.write(dom_3)
            st.write(dom_4)
            st.write(dom_5)
            col1, col2 ,col3, col4,col5= st.columns(5)
            with col1:
                 color1 = st.color_picker('Color 1','#'+dom_1,key=1)
                 st.write(f"{color1}")
            with col2:
                 color2 = st.color_picker('Color 2','#'+dom_2,key=2)
                 st.write(f"{color2}")
            with col3:
                 color3 = st.color_picker('Color 3','#'+dom_3,key=3)
                 st.write(f"{color3}")
            with col4:
                 color4 = st.color_picker('Color 4','#'+dom_4,key=4)
                 st.write(f"{color4}")
            with col5:
                 color5 = st.color_picker('Color 5','#'+dom_5,key=5)
                 st.write(f"{color5}")
                 
            #ask user to select 2 colors for recommendation
            st.write('Select exactly 2 colors below for new color recommendation:')
            option_1 = st.checkbox('Color 1')
            option_2 = st.checkbox('Color 2')
            option_3 = st.checkbox('Color 3')
            option_4 = st.checkbox('Color 4')
            option_5 = st.checkbox('Color 5')
            known_variables = option_1 + option_2 + option_3 + option_4 + option_5
    
            if known_variables <2:
                st.write('You have to select 2 colors.')
            elif known_variables == 2:
                #give recommendations
                st.write("Calculating Color Recommendations...")
                #do recommendation here
                new_colors=[]
                if option_1:
                    new_colors.append(f"{color1}")
                if option_2:
                    new_colors.append(f"{color2}")
                if option_3:
                    new_colors.append(f"{color3}")
                if option_4:
                    new_colors.append(f"{color4}")
                if option_5:
                    new_colors.append(f"{color5}")
                sug_colors=color_rec(new_colors)        
                col1, col2= st.columns(2)
                with col1:
                    color1 = st.color_picker('Color 1','#'+str(sug_colors[0]),key=10)
                    st.write(f"{color1}")
                with col2:
                    color2 = st.color_picker('Color 2','#'+str(sug_colors[1]),key=11)
                    st.write(f"{color2}")
            #ask user to select the color to change
            st.write('Select the color you want to change the object to be:')
            option_10 = st.checkbox('Color #1')
            option_11 = st.checkbox('Color #2')
            known_variables = option_10 + option_11
    
            if known_variables <1:
                st.write('You have to select a color.')
                check2=False
            elif known_variables == 1:
                #store new color
                st.write("Check the box below to change color")
                if option_10:
                    new_color=f"{color1}"
                    new_color=hex_to_rgb(new_color)
                    new_color=np.append(new_color,'255')
                    new_color=tuple(new_color)                    
                elif option_11:
                    new_color=f"{color2}"
                    new_color=hex_to_rgb(new_color)
                    new_color=np.append(new_color,'255')
                    new_color=tuple(new_color)
            else:
                st.write('Only select 1 color.')
                check2=False
            
        check2=st.checkbox("Check this Box to change colors.")
        if check2:
            
            #recolor image
            st.write("Image Recoloring...")
            #mask_1=mask_rect(image,int(left*scalex),int(top*scaley),int(width*scalex),int(height*scaley))
            mask_1=cv.cvtColor(mask_1, cv.COLOR_BGR2RGBA)
            new_image=color_change(mask_1,original,new_color)
             
            #output recolored image
            st.image(new_image, channels="RGB", width=int(w))
    
