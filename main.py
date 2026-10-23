import os
import sys
import subprocess
import time

def clear_screen():
    # Clear the terminal screen in a cross-platform manner.
    os.system("cls" if os.name == "nt" else "clear")

def welcome_message():
    clear_screen()
    # ASCII art banner for a creative start.
    banner = r'''
 _       __       __                            
| |     / /___   / /_____ ____   ____ ___   ___ 
| | /| / // _ \ / // ___// __ \ / __ `__ \ / _ \
| |/ |/ //  __// // /__ / /_/ // / / / / //  __/
|__/|__/ \___//_/ \___/ \____//_/ /_/ /_/ \___/ 
    '''
    print(banner)
    
    greetings = [
        "Network Recon and Detection Tool",
        "Your one-stop solution for network recon and intrusion detection using Machine Learning.",
        "Let's secure your network with these tools."
    ]
    
    for line in greetings:
        for char in line:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.03)
        print()
        time.sleep(0.5)
    print()

def run_tool(module):
    """Run one tool as a child process (``python -m <module>``) from the project folder,
    so its memory (a large model bundle) is freed and Ctrl+C or a crash in it brings the
    menu back."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        subprocess.run([sys.executable, "-m", module], cwd=base_dir)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"Error running {module}: {e}")

SCANNER = "toolkit.recon.cli"
CLI_IDS = "toolkit.ids_cli"
WEB_IDS = "toolkit.web"


def ask_yes_no(question):
    """Ask until the answer is 'y' or 'n'; return True for 'y'."""
    while True:
        answer = input(question).strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        print("Please enter 'y' or 'n'.")


def main_menu():
    # Flat loop: every sub-menu returns here instead of calling back into the menus,
    # so the call stack stays the same however many times a tool is run.
    while True:
        welcome_message()
        print("1. Network Recon")
        print("2. Intrusion Detection (CLI)")
        print("3. Intrusion Detection (Web)")
        print("4. Exit")
        choice = input("Enter your choice [1-4]: ").strip()
        if choice == "1":
            run_tool(SCANNER)
            if intrusion_prompt():
                intrusion_menu()
        elif choice == "2":
            run_tool(CLI_IDS)
            if prompt_repeat_detection():
                intrusion_menu()
        elif choice == "3":
            run_tool(WEB_IDS)
            if prompt_repeat_detection():
                intrusion_menu()
        elif choice == "4":
            print("Exiting the program. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

def intrusion_prompt():
    """After a recon run: True to go on to intrusion detection, False for the main menu."""
    if ask_yes_no("Proceed to Intrusion Detection? (y/n): "):
        return True
    print("\nReturning to Main Menu...")
    time.sleep(1)
    return False

def intrusion_menu():
    """Run detection tools until the user returns to the main menu."""
    while True:
        clear_screen()
        print("Intrusion Detection Options:")
        print("1. CLI Detection (CIC-IDS2017 model)")
        print("2. Web Detection (Dynamic Trainer)")
        print("3. Return to Main Menu")
        choice = input("Enter your choice [1-3]: ").strip()
        if choice == "1":
            run_tool(CLI_IDS)
        elif choice == "2":
            run_tool(WEB_IDS)
        elif choice == "3":
            return
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)
            continue
        if not prompt_repeat_detection():
            return

def prompt_repeat_detection():
    """After a detection run: True to run another detection, False for the main menu."""
    if ask_yes_no("\nDo you want to perform another detection? (y/n): "):
        return True
    print("\nReturning to Main Menu...")
    time.sleep(1)
    return False

if __name__ == "__main__":
    main_menu()
