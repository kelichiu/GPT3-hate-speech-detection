library(tidyverse)

zero_shot_misspelling <- read.csv(here::here("outputs", "data", "zero_shot_misspelling.csv"))
one_shot_misspelling <- read.csv(here::here("outputs", "data", "one_shot_misspelling.csv"))

# Zero-shot
# Summary of results
zero_shot_misspelling <- zero_shot_misspelling %>% separate(prompt, c(NA, "comment", NA), sep = "\"")
zeroshot_summary_race <- zero_shot_misspelling %>% 
  count(category, comment_id, answer) %>% 
  count(category, comment_id) %>% 
  filter(n > 1 & category == "racist")
zeroshot_summary_sex <- zero_shot_misspelling %>% 
  count(category, comment_id, answer) %>% 
  count(category, comment_id) %>% 
  filter(n > 1 & category == "sexist")

# Racist
zero_results_summary <- zero_shot_misspelling %>% 
                          filter(category == "racist" & comment_id %in% zeroshot_summary_race$comment_id) %>% 
                          filter(comment_id != 26) %>% # 'No,' vs. 'No.' for this comment
                          arrange(comment_id)

results <- tibble()
for (i in unique(zero_results_summary$comment_id)) {
  original <- zero_results_summary %>% filter(comment_id == i & status == "unedited")
  original_answer <- original["answer"] %>% as.character()
  edits <- zero_results_summary %>% filter(comment_id == i & status == "edited" & answer != original_answer)
  results <- rbind(results, original, edits)
}
results <- results %>% select(label, status, comment, answer)


  
# Construct summary table


# Sexist
zero_shot_misspelling %>% 
  filter(category == "sexist" & comment_id %in% zeroshot_summary_sex$comment_id)%>% 
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
