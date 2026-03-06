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

def run_tool(relative_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "tools", relative_path)
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        return
    try:
        subprocess.run([sys.executable, script_path])
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"Error running {script_path}: {e}")

def main_menu():
    while True:
        welcome_message()
        print("1. Network Recon")
        print("2. Intrusion Detection (CLI)")
        print("3. Intrusion Detection (Web)")
        print("4. Exit")
        choice = input("Enter your choice [1-4]: ").strip()
        if choice == "1":
            run_tool("Port_scan/Port_Scan.py")
            intrusion_prompt()
        elif choice == "2":
            run_tool("IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py")
            prompt_repeat_detection()
        elif choice == "3":
            run_tool("IDS/IDS_Web_CIC_Dynamic/app.py")
            prompt_repeat_detection()
        elif choice == "4":
            print("Exiting the program. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

def intrusion_prompt():
    while True:
        cont = input("Proceed to Intrusion Detection? (y/n): ").strip().lower()
        if cont == "y":
            intrusion_menu()
            break
        elif cont == "n":
            print("Exiting. Thank you!")
            sys.exit(0)
        else:
            print("Please enter 'y' or 'n'.")

def intrusion_menu():
    while True:
        clear_screen()
        print("Intrusion Detection Options:")
        print("1. CLI Detection (CIC-IDS2017 model)")
        print("2. Web Detection (Dynamic Trainer)")
        print("3. Return to Main Menu")
        choice = input("Enter your choice [1-3]: ").strip()
        if choice == "1":
            run_tool("IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py")
            prompt_repeat_detection()
            break
        elif choice == "2":
            run_tool("IDS/IDS_Web_CIC_Dynamic/app.py")
            prompt_repeat_detection()
            break
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

def prompt_repeat_detection():
    """After a detection run, ask the user if they want to repeat."""
    while True:
        rpt = input("\nDo you want to perform another detection? (y/n): ").strip().lower()
        if rpt == "y":
            intrusion_menu()
            return
        elif rpt == "n":
            print("\nReturning to Main Menu...")
            time.sleep(1)
            return
        else:
            print("Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main_menu()
