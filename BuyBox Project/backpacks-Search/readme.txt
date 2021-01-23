search_resultsX.json-- contains the search result for the corresponsing query. x denotes the first page or second page or third page of the result.

info-- This directory contains the product page information for each of the products that appear in the Search Engine Result Pages (SERPs). The seller information contained in the csv files for different products is actually the winner of the buy-box at that particular instant.

buybox-- This directory contains the information of sellers competing for a product out of which one had won the buy-box (information in corresponding file in info directory).

Name of each file in info and buybox folder is the ASIN (Amazon Specified Identification Number) of the corresponding product.

Note that buybox contains only those products for which number of sellers were more than 1.

Information in buybox files:  (tab separated)
Price + Delivery, Condition, Seller information, Delivery information
(for each of the sellers competing for the product)

Information in info files:    (tab separated)
ASIN, titleName, reviews, rating, brand, price, amazon choice keyword, seller, selller ratings, description (if any)