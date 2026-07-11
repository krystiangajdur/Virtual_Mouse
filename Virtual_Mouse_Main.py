from Virtual_Mouse_App import VirtualMouseApp

if __name__ == "__main__":
    app = VirtualMouseApp(model_path='svm_hand_model.pkl')
    app.run()