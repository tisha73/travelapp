show databases;
use inbox;
CREATE TABLE passenger1 (
    usercode INT PRIMARY KEY ,  
    company VARCHAR(255),                    
    name VARCHAR(255),                       
    gender ENUM('Male', 'Female', 'Other')   
);
CREATE TABLE flight (
    travelcode INT PRIMARY KEY, 
    user_id INT,
    departure VARCHAR(100),
    arrival VARCHAR(100),
    flight_type VARCHAR(50),
    flight_price DECIMAL(10, 2),
    flight_duration INT, 
    flight_distance INT, 
    flight_agency VARCHAR(100),
    departure_date DATETIME,
    FOREIGN KEY (user_id) REFERENCES passenger(usercode)
);
CREATE TABLE hotel (
    user_id INT,
    travel_code INT ,
    hotel_name VARCHAR(100),
    arrival_place VARCHAR(100),
    hotel_stay INT, 
    hotel_per_day_rent DECIMAL(10, 2),
    check_in DATETIME,
    hotel_total_price DECIMAL(10, 2),
    FOREIGN KEY (travel_code) REFERENCES flight(travelcode)
);
CREATE TABLE guest_profile (
    Guest_Id INT PRIMARY KEY,
    TravelCode INT,
    Guest_Name VARCHAR(100),
    Guest_Gender VARCHAR(10),
    Age INT,
    Guest_PhoneNo VARCHAR(15),
    Guest_Email VARCHAR(100),
    IdProof VARCHAR(50),
    FOREIGN KEY (TravelCode) REFERENCES flight(TravelCode)
);
ALTER TABLE guest_profile MODIFY Guest_PhoneNo VARCHAR(50);
ALTER TABLE guest_profile MODIFY Guest_Id INT;
ALTER TABLE passenger1 MODIFY gender VARCHAR(10);
SELECT * FROM review;
ALTER TABLE guest_profile DROP PRIMARY KEY;
DROP TABLE flight;
DROP TABLE hotel;
DROP TABLE guest_profile;
CREATE TABLE car_rent (
    Rent_ID INT AUTO_INCREMENT PRIMARY KEY,
    User_ID INT,  -- Reference to the user table
    TravelCode INT,  -- Reference to the flights table
    Rent_Date DATETIME,
    Pickup_Location VARCHAR(255) NOT NULL,
    Dropoff_Location VARCHAR(255) NOT NULL,
    Car_Type VARCHAR(100) NOT NULL,
    Rental_Agency VARCHAR(100) NOT NULL,
    Rental_Duration INT NOT NULL,  -- Duration in days
    Car_Total_Distance DECIMAL(10,2) NOT NULL,  -- Distance in km
    Fuel_Policy VARCHAR(50) NOT NULL,  -- E.g., Full-to-Full, Prepaid, etc.
    Car_BookingStatus VARCHAR(50) NOT NULL,  -- E.g., Confirmed, Pending, Cancelled
    Total_Rent_Price DECIMAL(10,2) NOT NULL,  -- Total rental cost

    -- Foreign Key Constraints
    FOREIGN KEY (User_ID) REFERENCES passenger(usercode),
    FOREIGN KEY (TravelCode) REFERENCES flight(travelcode)
);
CREATE TABLE Review (
    Review TEXT
)

DROP TABLE Review;

CREATE TABLE Car_review (
    Review TEXT
)

SELECT * from Car_review
Drop Table Test_car