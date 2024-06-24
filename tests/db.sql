CREATE TABLE Users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL
);

CREATE TABLE Orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    user_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES Users(id)
);

CREATE TABLE Categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

CREATE TABLE Products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES Categories(id)
);

CREATE TABLE OrderDetails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER NOT NULL,
    FOREIGN KEY (order_id) REFERENCES Orders(id),
    FOREIGN KEY (product_id) REFERENCES Products(id)
);


INSERT INTO Users (name, email) VALUES ('John Doe', 'john@example.com');
INSERT INTO Users (name, email) VALUES ('Jane Smith', 'jane@example.com');
INSERT INTO Users (name, email) VALUES ('Alice Johnson', 'alice@example.com');
INSERT INTO Users (name, email) VALUES ('Bob Brown', 'bob@example.com');

INSERT INTO Categories (name) VALUES ('Electronics');
INSERT INTO Categories (name) VALUES ('Books');
INSERT INTO Categories (name) VALUES ('Clothing');
INSERT INTO Categories (name) VALUES ('Furniture');

INSERT INTO Products (name, price, category_id) VALUES ('Smartphone', 699.99, 1);
INSERT INTO Products (name, price, category_id) VALUES ('Laptop', 999.99, 1);
INSERT INTO Products (name, price, category_id) VALUES ('E-book Reader', 129.99, 2);
INSERT INTO Products (name, price, category_id) VALUES ('T-shirt', 19.99, 3);
INSERT INTO Products (name, price, category_id) VALUES ('Jeans', 49.99, 3);
INSERT INTO Products (name, price, category_id) VALUES ('Dining Table', 299.99, 4);

INSERT INTO Orders (date, user_id) VALUES ('2024-06-01', 1);
INSERT INTO Orders (date, user_id) VALUES ('2024-06-02', 2);
INSERT INTO Orders (date, user_id) VALUES ('2024-06-03', 3);
INSERT INTO Orders (date, user_id) VALUES ('2024-06-04', 4);

INSERT INTO OrderDetails (order_id, product_id, quantity) VALUES (1, 1, 1); -- John Doe ordered 1 Smartphone
INSERT INTO OrderDetails (order_id, product_id, quantity) VALUES (1, 4, 2); -- John Doe ordered 2 T-shirts
INSERT INTO OrderDetails (order_id, product_id, quantity) VALUES (2, 2, 1); -- Jane Smith ordered 1 Laptop
INSERT INTO OrderDetails (order_id, product_id, quantity) VALUES (2, 6, 1); -- Jane Smith ordered 1 Dining Table
INSERT INTO OrderDetails (order_id, product_id, quantity) VALUES (3, 3, 1); -- Alice Johnson ordered 1 E-book Reader
INSERT INTO OrderDetails (order_id, product_id, quantity) VALUES (4, 5, 3); -- Bob Brown ordered 3 Jeans
