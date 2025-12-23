for (const driverKey in driver) {
const driverValue = driver[driverKey as keyof DriverVector] || 0;
const mapping = this.weights[driverKey];
