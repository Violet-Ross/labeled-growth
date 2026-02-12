library(DBI)
library(RMariaDB)
library(tidyverse)

# Reading in general (main.sql)
con <- dbConnect(
  MariaDB(),
  user = "root",
  password = "",
  host = "localhost",
  dbname = "dblpgeneral"
)

general <- dbReadTable(con, "general")

dbDisconnect(con)

write_csv(general, "throughput/general.csv")

# Reading in author_gender
con <- dbConnect(
  MariaDB(),
  user = "root",
  password = "",
  host = "localhost",
  dbname = "dblpauthor_gender"
)

genauth_old <- dbReadTable(con, "genauth_old")

dbDisconnect(con)

write_csv(genauth_old, "throughput/author_gender.csv")


# Reading in authors
con <- dbConnect(
  MariaDB(),
  user = "root",
  password = "",
  host = "localhost",
  dbname = "dblpauthors"
)

authors <- dbReadTable(con, "authors")

dbDisconnect(con)

write_csv(authors, "throughput/authors.csv")


