
export const createUserSlice = (set) => ({
    user: null,
    addUser: (user) => set(() => ({
        user: user
    })),
    logOut: (user) => set(() => ({
        user: user
    })),
})