'''
Code to scrape Amazon related item recommendations.
By: adash
'''
import sys
import json
#from importlib import reload
reload(sys)
sys.setdefaultencoding('utf8')
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import traceback
import os
from os import listdir
from os.path import isfile, join
import random
#import requests
from pyvirtualdisplay import Display
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.action_chains import ActionChains
from datetime import datetime


os.system('chmod +x ./Driver/geckodriver')
if not os.path.exists("./info"):
	os.makedirs("./info")
if not os.path.exists("./recommendation"):
	os.makedirs("./recommendation")
if not os.path.exists("./visitedURLs"):
	os.makedirs("./visitedURLs")
if not os.path.exists("./buybox"):
	os.makedirs("./buybox")

class Dictlist(dict):
	def __setitem__(self, key, value):
		try:
			self[key]
		except KeyError:
			super(Dictlist, self).__setitem__(key, [])
		self[key].append(value)

def selectProfile(driver, profileName):
	profiles = driver.find_elements_by_class_name("profile-icon")
	users = driver.find_elements_by_class_name("profile-name")
	if profiles is not None:
		for i in range(len(profiles)):
			if users[i].text == profileName:
				profiles[i].click()
				break

def authentication1(driver, loginUser):
	try:
		element = driver.find_element_by_name("email")
	except:
		element = driver.find_element_by_class_name("ui-text-input")
	element.send_keys(loginUser)
	submit = driver.find_element_by_class_name("a-button-input")
	submit.click()
	'''element2 = driver.find_element_by_name("password")
	element2.send_keys(passwordUser)
	submit2 = driver.find_element_by_css_selector(".btn.login-button.btn-submit.btn-small")
	submit2.click()'''

def authentication2(driver, loginUser):
	try:
		element = driver.find_element_by_name("password")
	except:
		element = driver.find_element_by_class_name("ui-text-input")
	element.send_keys(loginUser)
	submit = driver.find_element_by_class_name("a-button-input")
	submit.click()

def BFS(driver, startingTitle):
	# Get already loaded URLs
	mypath = 'visitedURLs/'
	products = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	loadedURLs = dict()
	for i in products:
		try:
			arq = open('info/'+i+'.tsv')
			name = arq.readline().split('\t')[1]
			arq.close()
			if name != '':
				loadedURLs[i] = True
		except:
			pass

	# * Dictionary which will keep the visited nodes
	visited = dict()

	# * Create a queue for BFS
	queue = []

	# * Mark the starting node as visited and enqueue it
	queue.append(startingTitle.split('/')[-1])
	visited[startingTitle.split('/')[-1]] = True

	while queue:
		# * Dequeue a vertex from queue
		nodeID = queue.pop(0)
		print(nodeID)
		try:
			if nodeID not in loadedURLs:
				pageLoaded = False
				while pageLoaded == False:
					try:
						driver.get('https://www.amazon.in/dp/' + nodeID)
						pageLoaded = True
					except:
						queue.append(nodeID)
						traceback.print_exc()
						print('Error loading page')
						nodeID = queue.pop(0)
						print (nodeID)
						pass
				print('Page loaded')
				time.sleep(random.randint(3,5))
			
				loadedOverview = False
				loadedRecommendation = False
				try:
					panel = driver.find_element_by_class_name('centerColAlign')
				except:
					panel = driver

				try:
					btn = driver.find_element_by_css_selector(".a-icon.a-icon-arrow.a-icon-small.arrow-icon")
				except:
					continue

				# Overview

				print ('Getting overview...')
				_original = False
				try:
					_titleName = panel.find_element_by_id('productTitle').text
					print(_titleName)
				except:
					try:
						_original = True
						_titleName = panel.find_element_by_class_name('logo').get_attribute("alt")
					except:
						pass
					if _original == False:
						_titleName = ''
					pass
				boolVal = False
				strings = ["headphone", "headphones", "earphone", "earphones", "earbud", "earbuds"]
				for s in strings:
					if s in _titleName.lower():
						boolVal = True
				if boolVal == False:
					continue
				try: 
					_delivery = driver.find_element_by_xpath("//div[@id='ddmDeliveryMessage']/span[@class='a-text-bold']").get_attribute('innerHTML')
					if ',' not in _delivery:
						_delivery = _delivery.split()
						_d1 = datetime.strptime(' '.join(_delivery[:2])+', 2020', "%b %d, %Y")
						_d2 = datetime.strptime(' '.join([_delivery[0], _delivery[-1]])+', 2020', "%b %d, %Y")
						_dnow = datetime.now()
						_diff1 = _d1 - _dnow
						#print(_diff1)
						_diff2 = _d2 - _dnow
						#print(_diff2)
						_delivery = str(_diff1.days) + '-' + str(_diff2.days) + ' days'
					else:

						_d1 = datetime.strptime(_delivery.split(', ')[-1]+', 2020', "%b %d, %Y")
						_dnow = datetime.now()
						_diff1 = _d1 - _dnow
						_delivery = str(_diff1.days) + ' days'
				except Exception as e:
					print(e)
					_delivery = ''
				try:
					_reviews = panel.find_element_by_id('acrCustomerReviewText').text
				except:
					_reviews = ''
					pass
				try:
					_rating = panel.find_element_by_id('acrPopover').get_attribute('title')
				except:
					_rating = ''
					pass
				try:
					_producer = panel.find_element_by_id('bylineInfo').text
				except:
					_producer = ''
					pass
				try:
					_price = panel.find_element_by_class_name('a-color-price').text
				except:
					_price = ''
					pass
				try:
					_achoice = panel.find_element_by_class_name('ac-badge-text-primary').text+panel.find_element_by_class_name('ac-badge-text-secondary').text
				except:
					_achoice = ''
					pass
				try:
					_seller = panel.find_element_by_id('shipsFromSoldBy_feature_div').text
				except:
					_seller = ''
					pass

				productsFile = open('info/'+nodeID+".tsv", 'w')
				productsFile.write(nodeID + '\t' + _titleName + '\t' + _delivery + '\t' + _reviews + '\t' + _rating + '\t' + _producer + '\t' + _price + '\t' + _achoice+ '\t' + _seller)
				productsFile.close()
				loadedOverview = True

				# Recommendations

				print ('Getting recommendations...')
				
				'''lists = list()
				relatedIDs = list()
				try:
					try:
						element = driver.find_element_by_id('sp_detail')
					except:
						element = driver.find_element_by_id('sp_detail2')
					relatedness = element.find_element_by_css_selector(".a-carousel-heading").text
					print(relatedness)
					driver.execute_script("arguments[0].scrollIntoView();", element)
					time.sleep(3)
					if element.find_element_by_css_selector(".a-carousel-page-current").text != '':
						current = int(element.find_element_by_css_selector(".a-carousel-page-current").text)
						max_panel = int(element.find_element_by_css_selector(".a-carousel-page-max").text)
					else:
						page_info = element.find_element_by_class_name("a-carousel-page-count").text
						if page_info != '':
							current = int(page_info.split()[1])
							max_panel = int(page_info.split()[-1])
						else:
							current = 1
							max_panel = 1
					if max_panel > 15:
						max_panel = 15
					while current <= max_panel :
						time.sleep(5)
						for el in element.find_elements_by_class_name("p13n-asin"):
							local_dict = eval(el.get_attribute("data-p13n-asin-metadata"))
							lists.append(local_dict['asin'])
						if max_panel > 1:
							action.move_to_element(element).perform()
							try:
								next_btn = element.find_element_by_css_selector(".a-icon.a-icon-next")#btn login-button btn-submit btn-small
								next_btn.click()
							except Exception as e:
								next_btn = element.find_element_by_css_selector(".a-link-normal.a-carousel-goto-nextpage.sp_pagination_button.sp_pagination_button.sp_pagination_right")
								next_btn.click()
															
						current += 1
					Reco = {}
					Reco[nodeID] = lists
					if relatedness == "Customers who viewed this item also viewed":
						relatedIDs = lists
						loadedRecommendation = True
					if not os.path.exists("./recommendation/"+relatedness):
						os.makedirs("./recommendation/"+relatedness)
					json.dump(Reco, open('recommendation/'+relatedness+'/'+nodeID+'.json', 'w'))

				except Exception as e:
					print(e)
					pass
				

				lists = list()
				try:
					element = driver.find_element_by_xpath("//*[contains(@id, 'desktop-dp-sims_purchase-similarities-')]")
					relatedness = element.find_element_by_css_selector(".a-carousel-heading").text
					print(relatedness)
					driver.execute_script("arguments[0].scrollIntoView();", element)
					time.sleep(3)
					if element.find_element_by_css_selector(".a-carousel-page-current").text != '':
						current = int(element.find_element_by_css_selector(".a-carousel-page-current").text)
						max_panel = int(element.find_element_by_css_selector(".a-carousel-page-max").text)
					else:
						page_info = element.find_element_by_class_name("a-carousel-page-count").text
						if page_info != '':
							current = int(page_info.split()[1])
							max_panel = int(page_info.split()[-1])
						else:
							current = 1
							max_panel = 1
					if max_panel > 15:
						max_panel = 15
					while current <= max_panel :
						time.sleep(5)
						for el in element.find_elements_by_class_name("p13n-asin"):
							local_dict = eval(el.get_attribute("data-p13n-asin-metadata"))
							lists.append(local_dict['asin'])
						if max_panel > 1:
							next_btn = element.find_element_by_css_selector(".a-icon.a-icon-next")#btn login-button btn-submit btn-small
							next_btn.click()
						current += 1
					Reco = {}
					Reco[nodeID] = lists
					if relatedness == "Customers who viewed this item also viewed":
						relatedIDs = lists
						loadedRecommendation = True
					if not os.path.exists("./recommendation/"+relatedness):
						os.makedirs("./recommendation/"+relatedness)
					json.dump(Reco, open('recommendation/'+relatedness+'/'+nodeID+'.json', 'w'))

				except Exception as e:
					print(e)
					pass'''


				lists = list()
				try:
					element = driver.find_element_by_xpath("//*[contains(@id, 'desktop-dp-sims_session-similarities-')]")
					relatedness = element.find_element_by_css_selector(".a-carousel-heading").text
					print(relatedness)
					driver.execute_script("arguments[0].scrollIntoView();", element)
					time.sleep(3)
					if element.find_element_by_css_selector(".a-carousel-page-current").text != '':
						current = int(element.find_element_by_css_selector(".a-carousel-page-current").text)
						max_panel = int(element.find_element_by_css_selector(".a-carousel-page-max").text)
					else:
						page_info = element.find_element_by_class_name("a-carousel-page-count").text
						if page_info != '':
							current = int(page_info.split()[1])
							max_panel = int(page_info.split()[-1])
						else:
							current = 1
							max_panel = 1
					if max_panel > 15:
						max_panel = 15
					while current <= max_panel :
						time.sleep(5)
						for el in element.find_elements_by_class_name("p13n-asin"):
							local_dict = eval(el.get_attribute("data-p13n-asin-metadata"))
							lists.append(local_dict['asin'])
						if max_panel > 1:
							next_btn = element.find_element_by_css_selector(".a-icon.a-icon-next")#btn login-button btn-submit btn-small
							next_btn.click()
						current += 1
					Reco = {}
					Reco[nodeID] = lists
					if relatedness == "Customers who viewed this item also viewed":
						relatedIDs = lists
						loadedRecommendation = True
					if not os.path.exists("./recommendation/"+relatedness):
						os.makedirs("./recommendation/"+relatedness)
					json.dump(Reco, open('recommendation/'+relatedness+'/'+nodeID+'.json', 'w'))

				except Exception as e:
					print(e)
					pass
				
				lists = list()
				try:
					btn = driver.find_element_by_css_selector(".a-icon.a-icon-arrow.a-icon-small.arrow-icon")
					btn.click()
					print('Getting buy box data')
					l1=[]
					price_delivery = driver.find_element_by_xpath("//div[@class='a-column a-span2'][1]/span").text.strip()
					l1.append(price_delivery)
					condition = driver.find_element_by_xpath("//div[@class='a-column a-span3'][1]/span").text
					l1.append(condition)
					seller_info = driver.find_element_by_xpath("//div[@class='a-column a-span2'][2]/span").text
					l1.append(seller_info)
					delivery = driver.find_element_by_xpath("//div[@class='a-column a-span3'][2]/span").text
					l1.append(delivery)
					buyBoxFile = open('buybox/'+nodeID+".tsv", 'w')
					buyBoxFile.write(price_delivery+'\t'+condition+'\t'+seller_info+'\t'+delivery+'\n')
					
					try:
						while 1:
							time.sleep(3)
							rows = driver.find_elements_by_xpath("//div[@class='a-row a-spacing-mini olpOffer']")
							#print([row.text for row in rows])
													
							for row in rows:
								l = []
								element = row.find_element_by_css_selector(".a-column.a-span2.olpPriceColumn")
								m = []
								price = element.find_element_by_xpath("span[contains(@class, 'a-size-large a-color-price olpOfferPrice a-text-bold')]/span").text
								m.append(price)
								prime = ''
								try:
									a = element.find_element_by_class_name("supersaver")
									prime = 'Prime : Yes'
								except Exception as e:
									prime = 'Prime : No'
								m.append(prime)
								delivery = driver.find_element_by_class_name("olpShippingInfo").text.split('.')[0]
								m.append(delivery)
								pay_on_delivery = element.text.split('\n')[-1]
								m.append(pay_on_delivery)
								l.append(m)
								
								element = row.find_element_by_css_selector(".a-column.a-span3.olpConditionColumn")
								m = []
								new = element.find_element_by_xpath("div[contains(@class, 'a-section a-spacing-small')]/span[contains(@class, 'a-size-medium olpCondition a-text-bold')]").text
								m.append(new)
								l.append(m)
								
								element = row.find_element_by_css_selector(".a-column.a-span2.olpSellerColumn")
								m = []
								seller_name = element.find_element_by_xpath("h3[contains(@class, 'a-spacing-none olpSellerName')]/span/a").text
								m.append(seller_name)
								try:
									star = element.find_element_by_xpath("p[contains(@class, 'a-spacing-small')]/i/span[@class = 'a-icon-alt']").get_attribute('innerHTML')
									#print(star)
									m.append(star)
									rating = element.find_element_by_xpath("p[contains(@class, 'a-spacing-small')]/a/b").text
									m.append(rating)
									no_rating = element.find_element_by_xpath("p[contains(@class, 'a-spacing-small')]").text.split('(')[-1][:-2]
									m.append(no_rating)
								except Exception as e:
									#print(e)
									just_launched = element.find_element_by_xpath("p[contains(@class, 'a-spacing-small')]/b[contains(@class, 'olpJustLaunched')]").text
									m.append(just_launched)
								l.append(m)
								
								element = row.find_element_by_css_selector(".a-column.a-span3.olpDeliveryColumn")
								m = []
								amazonFulfilled = ''
								try:
									a = element.find_element_by_xpath("div[contains(@class, 'olpBadgeContainer')]")
									amazonFulfilled = 'Fullfilled by Amazon'
								except Exception as e:
									#print(e)
									amazonFulfilled = 'Not fulfilled by Amazon'
								m.append(amazonFulfilled)
								txt = element.text.split('\n')[0]
								if txt != '' and 'Arrives between' in txt:
									txt = txt.replace('Arrives between ', '')
									txt = txt.replace('.', '')
									#print(txt)
									t = txt.split(' - ')
									if t[0]==txt:
										txt = txt.split('-')
										try:
											_d1 = datetime.strptime(txt[0]+', 2020', "%B %d, %Y")
										except:
											_d1 = datetime.strptime(txt[0]+', 2020', "%b %d, %Y")
										try:
											_d2 = datetime.strptime(' '.join([txt[0].split()[0], txt[-1]])+', 2020', "%B %d, %Y")
										except:
											_d2 = datetime.strptime(' '.join([txt[0].split()[0], txt[-1]])+', 2020', "%b %d, %Y")
										_dnow = datetime.now()
										_diff1 = _d1 - _dnow
										#print(_diff1)
										_diff2 = _d2 - _dnow
										#print(_diff2)
										txt = str(_diff1.days) + '-' + str(_diff2.days) + ' days'
									else:
										t[0] = t[0].split(', ')[-1]
										try:
											_d1 = datetime.strptime(t[0] + ', 2020', "%b %d, %Y")
										except:
											_d1 = datetime.strptime(t[0] + ', 2020', "%B %d, %Y")
										t[1] = t[1].split(', ')[-1]
										try:
											_d2 = datetime.strptime(t[1] + ', 2020', "%b %d, %Y")
										except:
											_d2 = datetime.strptime(t[1] + ', 2020', "%B %d, %Y")
										_dnow = datetime.now()
										_diff1 = _d1 - _dnow
										#print(_diff1)
										_diff2 = _d2 - _dnow
										#print(_diff2)
										txt = str(_diff1.days) + '-' + str(_diff2.days) + ' days'
									m.append(txt)
								l.append(m)
								lists.append(l)
							try: 
								nxt = driver.find_element_by_xpath("//div[contains(@class, 'a-text-center a-spacing-large')]/ul[contains(@class, 'a-pagination')]/li[contains(@class, 'a-last')]/a")
								#print(1)
								nxt.click()
							except Exception as e:
								#print(e)
								break
							
					except Exception as e:
						#print("breaking")
						print(e)
						#break	
					#print(lists)	
					for i in lists:	
						newRow = ''
						text =[]
						for j in i:
							text.append(' | '.join(j))
						newRow = '\t'.join(text)
						buyBoxFile.write(newRow+'\n')
					buyBoxFile.close()
							
				except Exception as e:
					print(e)
					pass
					
				'''
				lists = list()
				try:
					element = driver.find_element_by_id("sims-consolidated-4_feature_div")
					relatedness = element.find_element_by_css_selector(".a-carousel-heading").text
					print(relatedness)
					driver.execute_script("arguments[0].scrollIntoView();", element)
					current = int(element.find_element_by_css_selector(".a-carousel-page-current").text)
					#print(current)
					max_panel = int(element.find_element_by_css_selector(".a-carousel-page-max").text)
					#print(max_panel)
					while current <= max_panel :
						time.sleep(10)
						#print(len(element.find_elements_by_class_name("p13n-asin")))
						for el in element.find_elements_by_class_name("p13n-asin"):
							local_dict = eval(el.get_attribute("data-p13n-asin-metadata"))
							lists.append(local_dict['asin'])
						#print(lists)
						#exit(0)
						next_btn = element.find_element_by_css_selector(".a-icon.a-icon-next")#btn login-button btn-submit btn-small
						next_btn.click()
						current += 1
					#print(len(lists))
					#print(lists)
					Reco = {}
					Reco[nodeID] = lists
					if relatedness == "Customers who viewed this item also viewed":
						relatedIDs = lists
						loadedRecommendation = True
					if not os.path.exists("./recommendation/"+relatedness):
						os.mkdirs("./recommendation/"+relatedness)
					json.dump(Reco, open('recommendation/'+relatedness+'/'+nodeID+'.json', 'w'))
				except:
					pass
				'''
				#exit(0)
				#print(relatedIDs)
				for relatedID in relatedIDs:
					if relatedID not in visited:
						queue.append(relatedID)


				if loadedOverview and loadedRecommendation:
					loadedURLs[nodeID] = True
					print (len(loadedURLs))
					arq = open(mypath + nodeID, 'w')
					arq.write('')
					arq.close()
				#exit(0)	
				time.sleep(random.randint(5,10))
					
			else:
				print ('Recovering mode...')

				recoverFile = json.load(open('recommendation/'+'Customers who viewed this item also viewed/' + nodeID + '.json'))
				recommendations = recoverFile[recoverFile.keys()[0]]
				for title in recommendations:
					if title not in visited:
						queue.append(title)
						visited[title] = True

		except Exception as e:
			print ('Error 1', e)
			time.sleep(10)
			pass
			
		if len(queue) == 0:
			return False
			

def main():
	
	profile = webdriver.FirefoxProfile()
	'''profile.set_preference('network.proxy.type',1)
	profile.set_preference('network.proxy.http',"172.16.2.30")
	profile.set_preference('network.proxy.https',"172.16.2.30")
	profile.set_preference("network.proxy.ssl", "172.16.2.30")
	profile.set_preference('network.proxy.http_port',8080)
	profile.set_preference('network.proxy.https_port',8080)
	profile.set_preference("network.proxy.ssl_port", 8080)
	profile.update_preferences()'''
	options = Options()
	options.add_argument("--headless")
	#options = Options()
	#options.headless = True
	options = webdriver.FirefoxOptions()
	options.headless = True
	binary = FirefoxBinary('/usr/bin/firefox')
	#driver = webdriver.Firefox(options=options)
	driver = webdriver.Firefox(executable_path = './Driver/geckodriver', firefox_profile=profile)
	#driver = webdriver.Firefox(executable_path = './Driver/geckodriver', firefox_profile=profile)
	#driver = webdriver.Firefox()
	#driver.set_window_size(1920,1080)
	driver.set_page_load_timeout(60)
	driver.get("https://www.amazon.in/ap/signin?_encoding=UTF8&ignoreAuthState=1&openid.assoc_handle=inflex&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.ns.pape=http%3A%2F%2Fspecs.openid.net%2Fextensions%2Fpape%2F1.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.amazon.in%2Fgp%2Fyourstore%2Fhome%3Fie%3DUTF8%26ref_%3Dnav_custrec_signin&switch_account=")	
	print ('Page loaded')
	
	
	# Authenticate
	inputfile = open('AuthenticationDetails.txt', 'r').readlines()
	mailID = inputfile[0].split('<||>')[1].strip()
	password = inputfile[1].split('<||>')[1].strip()
	authentication1(driver, mailID)
	authentication2(driver, password)

	print('Signed In')
	#exit(0)
	time.sleep(5)
	print ('Authenticated')
	#driver.save_screenshot("screen.png")

	#value = BFS(driver, 'https://www.amazon.in/dp/B079LNSZQV')
	#value = BFS(driver, 'https://www.amazon.in/dp/B01J82IYLW')
	value = BFS(driver, 'https://www.amazon.in/dp/B07HKVCVSY')
	#value = BFS(driver, 'https://www.amazon.in/dp/046509760X')
	driver.close()
	driver.quit()
	
	return value


if __name__ == "__main__":
	error = True
	while error:
		try:
			error = main()
			if error == True:
				print ('Error loading all the files, starting recovery')
		except Exception as e:
			print (e)
			print ('Starting recovery')
 
