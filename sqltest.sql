WITH SELLERALERTS AS 
( 
  SELECT S.ID AS ID, COUNT(I.INCIDENTID) AS NALERTS
  FROM SELLER AS S 
	INNER JOIN INCIDENT I ON S.ID = I.CUSTOMERID
  WHERE I.DETECTIONLINKID IN (5,6,7,8) OR I.CONTENTLINKID IN (5,6,7,8)
  GROUP BY S.ID
), 

SELLERINFRIG AS 
( 
  SELECT S.ID AS ID, COUNT(I.INCIDENTID) AS NINFRIG
  FROM SELLER AS S 
	INNER JOIN INCIDENT I ON S.ID = I.CUSTOMERID
  WHERE I.DETECTIONLINKID IN (15,19) OR I.CONTENTLINKID IN (15,19)
  GROUP BY S.ID
),

SELLERDOMAIN AS
(
  SELECT S.ID AS ID, DP.NAME AS DOMAINNAME
  FROM SELLER AS S
	INNER JOIN DOMAIN AS D ON S.DOMAINID = D.DOMAINID
	INNER JOIN DOMAINCONFIGURATION AS DC ON D.DOMAINCONFIGURATIONID = DC.DOMAINCONFIGURATIONID
	INNER JOIN DOMAINPLATFORM AS DP ON D.DOMAINPLATFORMID = DP.DOMAINPLATFORMID
),

SELLERGROUP AS
(
  SELECT S.SELLERGROUPID as GROUPID, 
	 MIN(G.CREATIONDATE) AS MINDATE, 
	 SUM(E.FLOATVALUE) AS TOTSTOCKS,
	 COUNT(S.ID) AS NSELLERS,
	 SUM(SI.ID) AS TOTINFRIG,
 	 SUM(SA.ID) AS TOTALERTS,
	 COUNT(SD.ID) AS NUMPLATFS
  FROM SELLER AS S 
  	INNER JOIN SELLERCOMPANYDATA AS G ON S.ID = G.SELLERID
	LEFT JOIN SELLERINFRIG AS SI ON S.ID = SI.ID
	LEFT JOIN SELLERALERTS AS SA ON S.ID = SA.ID
	LEFT JOIN SELLERDOMAIN AS SD ON S.ID = SD.ID
  WHERE E.EXTRAINFOTYPE = 2
  GROUP BY S.SELLERGROUPID
),

OLDEST AS
(
  SELECT S.ID AS ID, S.NAME AS GNAME 
  FROM SELLERS S
	INNER JOIN SELLERCOMPANYDATA AS G ON S.ID = G.SELLERID 
  WHERE (S.SELLERGROUPID, G.CREATIONDATE) IN (SELECT GROUPID, MINDATE FROM SELLERGROUP)
) AS OLDEST

SELECT S.NAME AS SellerName, 
	O.GNAME AS SellerGroupName, 
	SG.NSELLERS AS GroupSize,
	DATEDIFF(day, S.SHOPOPENDATE, getdate()) AS Age,
	CASE WHEN (SG.NSELLERS IS NOT 0) THEN CONCAT(D.DOMAINNAME,' (',SG.NUMPLATFS,')') ELSE D.DOMAINNAME END AS Platforms
	SA.NALERTS AS SellerAlerts,
	SG.TOTSALERTS AS GroupAlerts,  
	SI.NINFRIG AS SellerInfringements,
	SG.TOTSINFRIG AS GroupInfringements,
 	E.FLOATVALUE AS SellerStocks,
	SG.TOTSTOCKS AS GroupStocks
	C.COUNTRY AS SellerCountry
FROM SELLERS AS S
	INNER JOIN SELLERCOMPANYDATA AS C ON S.ID = C.SELLERID
	INNER JOIN SELLERDOMAIN AS D ON S.ID = D.ID
	LEFT JOIN SELLERGROUP ON SG.GROUPID = S.SELLERGROUPID
	LEFT JOIN SELLERALERTS AS SA ON S.ID = SA.ID
	LEFT JOIN SELLERINFRIG AS SI ON S.ID = SI.ID 
	LEFT JOIN EXTRAINFO AS E ON S.LINKD = E.LINKD
	LEFT JOIN OLDEST AS O ON O.ID = S.ID
WHERE E.EXTRAINFOTYPE = 2 



