import { createUserSlice } from "./User/userSlice";
import { create} from 'zustand';
import {persist, createJSONStorage} from 'zustand/middleware'

const useStore = create(
    persist(
        (...a) => ({
        ...createUserSlice(...a)
        }),
        {
            name:'user-data',
            storage: createJSONStorage(() => sessionStorage)
        }
    )
)

export default useStore;