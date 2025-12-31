# ================== PRICE CALCULATOR ==================
elif app_mode == "🧮 Price Calculator":
    st.title("🧮 Car Price Calculator")

    st.write("You can either enter car details manually or select a car to get historical insights.")

    # ---- Car Name Selection ----
    car_name = st.selectbox(
        "Select Car Name (optional, leave blank to enter manually)",
        [""] + sorted(df_raw["Car_Name"].unique())
    )

    if car_name:  # User selected a car
        car_df = df_raw[df_raw["Car_Name"] == car_name]
        avg_year = int(car_df["Year"].mean())
        avg_kms = int(car_df["Kms_Driven"].mean())
        avg_owner = int(car_df["Owner"].mode()[0])
        avg_price = float(car_df["Selling_Price"].mean())

        st.info(f"📊 Historical Average Price for {car_name}: {avg_price:.2f} Lakhs")
    else:  # Manual input default values
        avg_year = 2020
        avg_kms = 30000
        avg_owner = 0
        avg_price = 5.0

    col1, col2 = st.columns(2)

    with col1:
        asking_price = st.number_input(
            "Asking Price (Lakhs)", 0.0, 50.0, round(avg_price, 2)
        )
        kms = st.number_input("Kilometers Driven", 0, 500000, avg_kms)
        car_age = st.slider("Car Age (Years)", 0, 25, 2025 - avg_year)

    with col2:
        owner = st.selectbox("Owner Type", [0, 1, 3], index=0)
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
        seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("🚀 Predict Price"):
        input_data = pd.DataFrame([{
            "Present_Price": asking_price,
            "Kms_Driven": kms,
            "Owner": owner,
            "Car_Age": car_age,
            "Fuel_Type_Diesel": 1 if fuel == "Diesel" else 0,
            "Selling_type_Individual": 1 if seller == "Individual" else 0,
            "Transmission_Manual": 1 if transmission == "Manual" else 0
        }])

        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        predicted_price = rf.predict(input_data)[0]

        st.success(f"💰 Predicted Fair Price: {predicted_price:.2f} Lakhs")

        # ---- Deal Decision ----
        if asking_price <= predicted_price:
            st.success("🟢 Reasonable Price / Good Deal")
        else:
            st.error("🔴 Overpriced Car")

        # ---- Visualization ----
        fig, ax = plt.subplots()
        ax.bar(
            ["Asking Price", "Predicted Price", "Historical Avg"],
            [asking_price, predicted_price, avg_price],
            color=['orange', 'green', 'blue']
        )
        ax.set_ylabel("Price (Lakhs)")
        ax.set_title(f"Price Comparison{' for ' + car_name if car_name else ''}")
        st.pyplot(fig)
