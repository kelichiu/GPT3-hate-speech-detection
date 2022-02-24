library(tidyverse)

zero_shot_misspelling <- read.csv(here::here("outputs", "data", "zero_shot_misspelling.csv"))
one_shot_misspelling <- read.csv(here::here("outputs", "data", "one_shot_misspelling.csv"))

# Zero-shot
# Summary of results
zero_shot_misspelling <- zero_shot_misspelling %>% separate(prompt, c(NA, "comment", NA), sep = "\"")
zeroshot_summary <- zero_shot_misspelling %>% count(category, comment_id, answer) %>% count(category, comment_id) %>% filter(n > 1)
zero_shot_misspelling %>% 
  filter(category == "racist" & comment_id %in% zeroshot_summary$comment_id)%>% 
  arrange(comment_id) %>%
  View()

# One-shot
oneshot_summary_race <- one_shot_misspelling %>% 
  count(category, comment_id, answer) %>% 
  count(category, comment_id) %>% 
  filter(n > 1 & category == "racist")
oneshot_summary_sex <- one_shot_misspelling %>% 
  count(category, comment_id, answer) %>% 
  count(category, comment_id) %>% 
  filter(n > 1 & category == "sexist")

# Racism results
one_shot_misspelling %>% 
  filter(category == "racist" & comment_id %in% oneshot_summary_race$comment_id)%>% 
  arrange(comment_id) %>%
  View()

# Sexism results
one_shot_misspelling %>% 
  filter(category == "sexist" & comment_id %in% oneshot_summary_sex$comment_id)%>% 
  arrange(comment_id) %>%
  View()
