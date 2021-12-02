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
    # change rgb values to hex values
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
        # print(temp)
        coor = colorsys.rgb_to_hsv(temp[0], temp[1], temp[2])
        # print(coor)
        if i == 0:
            new = coor
            RGB = temp
        else:
            # print(RGB)
            new = np.vstack((new, coor))
            RGB = np.vstack((RGB, temp))
        i = i + 1
        # pick top two colors. Offer recommendations based on analogous triadic and tetradic
    angle1 = new[0][0]
    angle2 = new[1][0]
    suggest = [[.0, .0, .0],
               [.0, .0, .0]]
    # first is the triadic recommendation
    # second is analogous

    # print(angle1)

    # print(angle2)

    dif = abs(angle1 - angle2)

    # print(dif)
    suggest[0][1] = (new[0][1] + new[1][1]) / 2
    suggest[1][1] = (new[0][1] + new[1][1]) / 2
    suggest[0][2] = (new[0][2] + new[1][2]) / 2
    suggest[1][2] = (new[0][2] + new[1][2]) / 2

    # print(suggest)

    if dif > 0.5:
        # print("big")
        this = abs(((new[0][0] + new[1][0]) / 2))
        that = abs(((new[0][0] + new[1][0]) / 2) + 0.5)
        suggest[0][0] = this
        suggest[1][0] = that
    elif dif < 0.5:
        # print("smol")
        this = abs(((new[0][0] + new[1][0]) / 2))
        that = abs(((new[0][0] + new[1][0]) / 2) + 0.5)
        if that >= 1:
            that = that - 1
        suggest[0][0] = that
        suggest[1][0] = this
    else:
        print("there is something awfuly wrong")
    # print(suggest)
    R, G, B = colorsys.hsv_to_rgb(suggest[0][0], suggest[0][1], suggest[0][2])
    other = [(np.int(np.round(R))), (np.int(np.round(G))), (np.int(np.round(B)))]
    fRGB = other
    # print(RGB)
    # print(fRGB)
    R, G, B = colorsys.hsv_to_rgb(suggest[1][0], suggest[1][1], suggest[1][2])
    other = [(np.int(np.round(R))), (np.int(np.round(G))), (np.int(np.round(B)))]
    fRGB = np.vstack((fRGB, other))

    # print(fRGB)

    hex_recs = rgb_to_hex((fRGB[0][0], fRGB[0][1], fRGB[0][2]))
    iHopeChrisDoesntReadThroughThisGodForsakenCode = rgb_to_hex((fRGB[1][0], fRGB[1][1], fRGB[1][2]))
    hex_recs = np.stack((hex_recs, iHopeChrisDoesntReadThroughThisGodForsakenCode))

    return hex_recs


@st.cache
def color_id(image, clusters):
    r = []
    g = []
    b = []
    # add all r,g,b data to array
    for row in image:
        for temp_r, temp_g, temp_b, temp in row:
            if temp != 0 or (temp_r != 0 or temp_g != 0 or temp_b != 0):
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)

    # make into dataframe
    image_df = pd.DataFrame({'red': r,
                             'green': g,
                             'blue': b})

    # standardize
    image_df['scaled_color_red'] = whiten(image_df['red'])
    image_df['scaled_color_blue'] = whiten(image_df['blue'])
    image_df['scaled_color_green'] = whiten(image_df['green'])

    # find clusters
    cluster_centers, _ = kmeans(image_df[['scaled_color_red',
                                          'scaled_color_blue',
                                          'scaled_color_green']], clusters)

    dominant_colors = []

    # find std dev
    red_std, green_std, blue_std = image_df[['red',
                                             'green',
                                             'blue']].std()

    # output dominant colors
    for cluster_center in cluster_centers:
        red_scaled, green_scaled, blue_scaled = cluster_center
        dominant_colors.append((
            int(red_scaled * red_std),
            int(green_scaled * green_std),
            int(blue_scaled * blue_std)

        ))

    return dominant_colors

@st.cache
def color_change(image, original, updated_color):
    #changes color of object to the new color
    image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
    original = cv.cvtColor(original, cv.COLOR_RGBA2RGB)
    final = original.copy()
    new_image = image.copy()

    new_image[np.where((image != [0, 0, 0]).all(axis=2))] = updated_color  # shape issue
    result = cv.addWeighted(new_image, 0.5, image, 0.3, 0)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    for row in range(np.shape(gray)[0]):
        for col in range(np.shape(gray)[1]):
            if gray[row][col] != 0:
                final[row][col][:] = result[row][col][:]
    # cv2.imshow('im', original)

    return final


@st.cache
def mask_rect(image, rect):
    # remove background with grabcut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # grabcut
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]

    return image

def mask_free(image, rect):
    # Object before using simple Rectangular Grabcut
    mask = np.zeros(image.shape[:2], np.uint8)  # initialize the layers for the rectangular mask
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    # establishes the mask using grabcut with a rectangle
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image1 = image * mask2[:, :, np.newaxis]  # saves a separate image for the user to view the simple grabcut
    if image1 is not None:
        cv.imwrite("image1.png", image1)

    # Using Free Drawings to create a Grabcut Mask with Streamlit
    # Create a canvas component
    # Allow users to draw on the original image to ensure all aspects are maintained

    st.write(
        "3. Draw red on the objects you want REMOVED from the object pictured below.")
    canvas_result2 = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#FF0000",
        background_color="#eeeeee",
        background_image=Image.open("image1.png"),
        update_streamlit=True,
        height=h, width=w,
        drawing_mode="freedraw",
        key="canvas2",
    )
    #submit = st.checkbox('Proceed')  # creates a streamlit button for users to proceed with the advanced grabcut
    column1, column2, column3 = st.columns([1,2,1])
    with column2:
        st.caption("Object Before Advanced Object Extraction\n\n")
    #if submit:  # (begin_free_grabcut == 'Yes'):
    newmask = canvas_result2.image_data  # Uses the drawing from the canvas for the new mask layer for the Grabcut
    # st.image(newmask, width=int(w))  # Displays the Drawn Data
    if newmask is not None:
        cv.imwrite("new_image_mask_grabcut.png", newmask)  # saves the newmask as a PNG
    newmask = cv.imread("new_image_mask_grabcut.png", 0)  # reads the new mask as a binary image
    newmask = cv.resize(newmask, (image.shape[1], image.shape[0]))  # Resizes the user drawn mask to match
    # the original scale of the image, in order to avoid losing image data

    # Overlay the User drawings on a gray background
    gray_background = Image.new(mode="RGB", size=(image.shape[1], image.shape[0]), color=90)
    gray_background.save("gray_background.png")
    gray_background = cv.imread("gray_background.png", 0)  # Problem Child
    newmask = cv.addWeighted(newmask, 1, gray_background, 1,
                                 0)  # Combines the gray background with the user drawn mask
    cv.imwrite("new_image_mask_grabcut.png", newmask)

    # Create the mask to be used for the grabcut
    # Where marked red (ensure background), change mask = 0
    # mask[newmask == 255] = 1  # Where marked white (ensure foreground), change mask = 1
    # However, this code is currently set to only use the ensure background
    for i in range(np.shape(newmask)[0]):
        for j in range(np.shape(newmask)[1]):
            if newmask[i][j] == 255:
                mask[i][j] = 1
            elif newmask[i][j] != 90 & newmask[i][j] != 255:
                mask[i][j] = 0
                
    mask, bgdModel, fgdModel = cv.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask[:, :, np.newaxis]

    # insert an st.stop() in an else statement so that the code does not continue until the user has verified they are done with their code

    return image


@st.cache
def background_only(image, original):
    # convert to RGBA
    picture = cv.cvtColor(image, cv.COLOR_BGR2RGBA)

    # Get the size of the image
    dimensions = picture.shape
    # define new color
    new_color = np.zeros(4)

    # Process every pixel, change black to transparent white,
    # adjust color of everything else
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            current_color = picture[i, j]
            if current_color[0] == 0 and current_color[1] == 0 and current_color[2] == 0 and current_color[3] == 255:
                new_color = original[i, j]
            else:
                new_color = (0, 0, 0, 0)
            picture[i, j] = (new_color)

    return picture


# main
st.title('Furniture Color Change App')

uploaded_img = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])  # JPEG dont work yet :(

if uploaded_img is not None:
    # change file into decoded image
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)

    # change original into RGBA
    original = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
    cv.imwrite("original.png", original)
    cv.imwrite("picture.png", image)

    #st.write("Choose the object that you would like to change...")
    st.write("1. Draw a SINGLE rectangle around the object that you would like to re-color.")

    # get dimensions and ratio of size, define size for canvas
    dimensions = image.shape
    ratio = dimensions[1] / dimensions[0]
    if ratio > 1:
        w = 600
        h = 600 / ratio
    else:
        h = 400
        w = 400 / ratio
    
    dim=(int(w),int(h))
    image=cv.resize(image,dim,interpolation=cv.INTER_AREA)
    original=cv.resize(original,dim,interpolation=cv.INTER_AREA)
    
    new_color=(1,0)
    
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
    step_counter = 3
    # define scale
    #scalex = dimensions[1] / w
    #scaley = dimensions[0] / h
    if canvas_result.json_data is not None:
        if canvas_result.json_data["objects"]!=[]:
            check = st.checkbox("Check this box to continue.")

            if check:
    
                # define rectangle object dimensions
                rect_obj = canvas_result.json_data["objects"][-1]
                left = rect_obj["left"]
                top = rect_obj["top"]
                width = rect_obj["width"]
                height = rect_obj["height"]
                rect = (int(left), int(top), int(width), int(height))
                grabcut_selection = st.radio("2. What method of object re-coloring would you like to proceed with?",
                                             ('Simple Object Extraction (Recommended)', 'Advanced Object Extraction'),
                                             help="If using 'Advanced Object Extraction', it may help to increase "
                                                  "the size of your rectangle")
                
                if grabcut_selection == 'Simple Object Extraction (Recommended)':
                    rectangle_mask = mask_rect(image, rect)
                    mask_1 = rectangle_mask
                    canvas_result.json_data["objects"] = []
                # Custom Grabcut with interactive foreground extraction
                elif grabcut_selection == 'Advanced Object Extraction':
                    # Begin Grabcut Free
                    free_draw_mask = mask_free(image, rect)
                    # Show the image after using grabcut free
                    if free_draw_mask is not None:
                        st.image(free_draw_mask, channels='BGR', width=int(w), caption="Object After Advanced Object Extraction")
                        mask_1 = free_draw_mask
                        canvas_result.json_data["objects"] = []
                    step_counter += 1
        
                # Pick your own color
                check1 = st.radio(str(step_counter) + ". How would you like to select the color of your object?", ('Pick my own color', 'Color Recommendation'),
                                  help="'Color Recommendation' will recommend two colors based on dominant colors "
                                       "in the room.")
                if check1 == 'Pick my own color':
                    color = st.color_picker('Pick A Color:', '#717171')
                    new_color = ImageColor.getcolor(color, "RGB")
                    step_counter += 1
                # color recommendation
                if check1 == 'Color Recommendation':
                    step_counter += 1
                    # st.write("Loading...")
        
                    # perform grabcut
                    # mask_1=mask_rect(image,int(left*scalex),int(top*scaley),int(width*scalex),int(height*scaley))
        
                    # make into background only
                    
                    background = background_only(mask_1, original)
                    
                    #st.image(background)  # REMOVE
                    # output dominant colors in columns
                    #st.write('Here are the dominant colors from the rest of the image.')
                    # ask user to select 2 colors for recommendation
                    st.write(str(step_counter) + '. Select exactly 2 colors below for new color recommendation:')
                    dom_color = color_id(background, 5)
                    dom_1 = str(rgb_to_hex(dom_color[0]))
                    dom_2 = str(rgb_to_hex(dom_color[1]))
                    dom_3 = str(rgb_to_hex(dom_color[2]))
                    dom_4 = str(rgb_to_hex(dom_color[3]))
                    dom_5 = str(rgb_to_hex(dom_color[4]))
                    dom_1 = dom_1.replace('-', '')
                    dom_2 = dom_2.replace('-', '')
                    dom_3 = dom_3.replace('-', '')
                    dom_4 = dom_4.replace('-', '')
                    dom_5 = dom_5.replace('-', '')
        
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        color1 = st.color_picker('', '#' + dom_1, key="C1")
                        option_1 = st.checkbox('', key=1)
                    with col2:
                        color2 = st.color_picker('', '#' + dom_2, key="C2")
                        option_2 = st.checkbox('', key=2)
                    with col3:
                        color3 = st.color_picker('', '#' + dom_3, key="C3")
                        option_3 = st.checkbox('', key=3)
                    with col4:
                        color4 = st.color_picker('', '#' + dom_4, key="C4")
                        option_4 = st.checkbox('', key=4)
                    with col5:
                        color5 = st.color_picker('', '#' + dom_5, key="C5")
                        option_5 = st.checkbox('', key=5, help="These colors are the dominant colors found within the rest of your image.")
        
                    # ask user to select 2 colors for recommendation
                    #st.write('Select exactly 2 colors above for new color recommendation:')
                    
                    known_variables = option_1 + option_2 + option_3 + option_4 + option_5
        
                    if known_variables != 2:
                        st.warning('You must select 2 colors.')
                    if known_variables == 2:
                        step_counter += 1
                        # give recommendations
                        #st.write("Calculating Color Recommendations...")
                        # ask user to select the color to change
                        st.write(str(step_counter) + '. Select the color you want to change the object to be:')
                        # do recommendation here
                        new_colors = []
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
                        sug_colors = color_rec(new_colors)
                        col1, col2 = st.columns(2)
                        with col1:
                            color1 = st.color_picker('', '#' + str(sug_colors[0]).replace('-', ''), key=6)
                            option_10 = st.checkbox('', key=7, help="Triadic Color Recommendation")
                        with col2:
                            color2 = st.color_picker('', '#' + str(sug_colors[1]).replace('-', ''), key=8)
                            option_11 = st.checkbox('', key=9, help="Analogous Color Recommendation")
                            
                        
                        
                        known_variables = option_10 + option_11
            
                        if known_variables == 1:
                            step_counter += 1
                            # store new color

                            if option_10:
                                new_color = f"{color1}"
                                new_color = hex_to_rgb(new_color)
                                new_color = tuple(new_color)
                            elif option_11:
                                new_color = f"{color2}"
                                new_color = hex_to_rgb(new_color)
                                new_color = tuple(new_color)
                            else:
                                new_color = (1, 0)
                    #else:
                        #st.write('You can only select 2 colors')
        
                if len(new_color) == 3:
                    columns1, columns2 = st.columns([1.25, 1])
                    with columns1:
                        st.write(str(step_counter) + ". Press this button to perform the object re-coloring:")
                    with columns2:
                        check2 = st.button('Re-Color', key="checkbox", help="If the final image is colored poorly, you may want to try using the 'Advanced Object Extraction'.")
        
                    if check2:
                        # recolor image
                        #st.write("Image Recoloring...")
                        # mask_1=mask_rect(image,int(left*scalex),int(top*scaley),int(width*scalex),int(height*scaley))
                        mask_1 = cv.cvtColor(mask_1, cv.COLOR_BGR2RGBA)
                        new_image = color_change(mask_1, original, new_color)

                        # output recolored image
                        st.image(new_image, channels="RGB", width=int(w))
                        cv.imwrite("object_recolor_image.png", new_image)
                        st.download_button("Download Image", "object_recolor_image.png", "Recolored_Object.png")
                        st.balloons()
                        #check3=st.checkbox("Check this Box when complete.")
                        #if check3:
                            #st.balloons()
            
                

