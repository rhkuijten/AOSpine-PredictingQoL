# Load the packages
library(eq5d)
library(readxl)
library(writexl)

# Define the input and output file paths
input_file_path <- "..."
output_file_path <- "..."

# Import the data
data <- read_excel(input_file_path)

# Initialize empty vectors to hold the results
B_eq5d_scores <- vector("numeric", length=nrow(data))
M3_eq5d_scores <- vector("numeric", length=nrow(data))

# Calculate the B eq5d score for each patient with complete B EQ5D items
for (i in 1:nrow(data)) {
  if (!any(is.na(c(data$B_Mob[i], data$B_Sel[i], data$B_Usu[i], data$B_Dis[i], data$B_Anx[i])))) {
    B_eq5d_scores[i] <- eq5d(scores=c(MO=data$B_Mob[i], SC=data$B_Sel[i], UA=data$B_Usu[i], PD=data$B_Dis[i], AD=data$B_Anx[i]), type="TTO", version="3L", country="Netherlands")
  } else {
    B_eq5d_scores[i] <- NA  # Assign NA for patients with incomplete B EQ5D items
  }
}

# Clear M3 values for patients who are dead at 3 months
for (i in 1:nrow(data)) {
  if (data[["3_months"]][i] == 0) { # If patient is dead at 3 months
    # Set M3 questionnaire responses to NA
    data$M3_Mob[i] <- NA
    data$M3_Sel[i] <- NA
    data$M3_Usu[i] <- NA
    data$M3_Dis[i] <- NA
    data$M3_Anx[i] <- NA
    data$M3_Date[i] <- NA
    data$M3_months[i] <- NA
  }
}

# Calculate the M3 eq5d score for each patient with complete M3 EQ5D items and alive at 3 months
for (i in 1:nrow(data)) {
  if (data[["3_months"]][i] == 1 && !any(is.na(c(data$M3_Mob[i], data$M3_Sel[i], data$M3_Usu[i], data$M3_Dis[i], data$M3_Anx[i])))) {
    M3_eq5d_scores[i] <- eq5d(scores=c(MO=data$M3_Mob[i], SC=data$M3_Sel[i], UA=data$M3_Usu[i], PD=data$M3_Dis[i], AD=data$M3_Anx[i]), type="TTO", version="3L", country="Netherlands")
  } else if (data[["3_months"]][i] == 0) {
    M3_eq5d_scores[i] <- 0 # Assign 0 for patients who are dead at three months
  } else {
    M3_eq5d_scores[i] <- NA  # Assign NA for patients with incomplete M3 EQ5D items
  }
}

# Add the eq5d scores to the data frame
data$B_Index <- B_eq5d_scores
data$M3_Index <- M3_eq5d_scores

# Export the data with the new scores
write_xlsx(data, output_file_path)
