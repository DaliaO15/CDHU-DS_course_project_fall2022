import requests
from bs4 import BeautifulSoup
import os 
import re

def trim_n_convert(image, rectangle):

# This fuction helps to trim dlib rectangles (since sometimes they can fall outside out the image),
# and to convert it to OpenCV boxes
# Created after https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/

	# Extract the vertices of the rectangle
	startX = rectangle.left()
	startY = rectangle.top()
	endX = rectangle.right()
	endY = rectangle.bottom()
	
	# Making the box to be inside the image 
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])

	# Compute the high and widht of the rectangle
	w = endX - startX
	h = endY - startY

	# Return new rectagle
	return (startX, startY, w, h)


def trim_n_convert_v2(image, epsilon, rectangle):
	# Extract the vertices of the rectangle
	startX = rectangle.left()
	startY = rectangle.top()
	endX = rectangle.right()
	endY = rectangle.bottom()

	# Making the box to be inside the image 
	startX = max(0, startX-epsilon)
	startY = max(0, startY-epsilon)
	endX = min(endX+epsilon, image.shape[1])
	endY = min(endY+epsilon, image.shape[0])

	# Compute the high and widht of the rectangle
	w = endX - startX
	h = endY - startY

	# Return new rectagle
	return (startX, startY, w, h)


def name_changer(root_url, page_start, page_end, folder):
    
    names2replace = os.listdir(folder) # List of the names of the images in the folder
    if '.DS_Store' in names2replace:
        names2replace.remove('.DS_Store')

    # Accessing every page 
    for page_nbr in range(page_start,page_end+1): # Loop through every page

        print(f'---- {page_nbr} ----') 
        
        # Modify the url to access each page in the search
        url = root_url + '&page=' + str(page_nbr)

        # Start the request
        r = requests.get(url=url)
        soup = BeautifulSoup(r.text, 'html.parser')
        photos = soup.find_all('div', {'class': 'teaser'})

        for item in photos: # Loop through every picture

            img_link = item.find('img')['src'] # Get the link to the image
            
            # Get ID and make in the image's name
            item_info = item.find('div', {'class': 'teaser__description aos-init'})
            id_ = item_info.find_all('p')[-1] # Accessing just the last pharagraph
            img_name = str(id_).replace('<p><strong>ID</strong> ','').replace('</p>','')
            # Note: some images have very problematic ID's so we must cleann them
            img_name = re.sub('[^A-Za-z0-9]+', '', img_name)
            img_name = img_name + '.jpg'

            if img_name in names2replace:
                img_name_new = img_link.split('zoom/')[-1]
                #img_name_new = img_name_new.split('.jpg')[0]
                img_name_new = img_name_new.replace('/','_')
                img_name_new = re.sub('[^A-Za-z0-9_.]+', '', img_name_new)
                os.rename(os.path.join(folder, img_name), os.path.join(folder, img_name_new)) # Replace the name