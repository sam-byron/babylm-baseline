#!/bin/bash
# filepath: run_training.sh

# Configuration
SCRIPT_PATH="transformer_trainer.py"
CONFIG_PATH="model_babylm_bert.json"  # Change this to your config file path
RESTART_DELAY=60  # seconds
MAX_RETRIES=10    # Maximum number of restart attempts (set to -1 for infinite)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run the training script
run_training() {
    local attempt=$1
    log "${GREEN}Starting training attempt #${attempt}...${NC}"
    
    # Run the Python script
    accelerate launch "$SCRIPT_PATH" --config_path "$CONFIG_PATH"
    return $?
}

# Main execution
main() {
    log "${GREEN}Training runner started${NC}"
    log "Script: $SCRIPT_PATH"
    log "Config: $CONFIG_PATH"
    log "Restart delay: ${RESTART_DELAY}s"
    log "Max retries: $MAX_RETRIES"
    echo "----------------------------------------"
    
    attempt=1
    
    while true; do
        # Run the training script
        run_training $attempt
        exit_code=$?
        
        # Check if script completed successfully
        if [ $exit_code -eq 0 ]; then
            log "${GREEN}Training completed successfully!${NC}"
            break
        else
            log "${RED}Training failed with exit code: $exit_code${NC}"
            
            # Check if we've reached max retries
            if [ $MAX_RETRIES -ne -1 ] && [ $attempt -ge $MAX_RETRIES ]; then
                log "${RED}Maximum retry attempts ($MAX_RETRIES) reached. Exiting.${NC}"
                exit 1
            fi
            
            # Wait before restarting
            log "${YELLOW}Waiting ${RESTART_DELAY} seconds before restart...${NC}"
            sleep $RESTART_DELAY
            
            attempt=$((attempt + 1))
            log "${YELLOW}Restarting training (attempt #${attempt})...${NC}"
        fi
    done
}

# Handle Ctrl+C gracefully
trap 'log "${RED}Received interrupt signal. Exiting...${NC}"; exit 130' INT TERM

# Run main function
main "$@"